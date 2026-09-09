/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * HT zero_copy config flag. Each test creates its own ep_group with the
 * mode under test; layout is FLAT with the shared 4-rank / 8-expert /
 * top-k=1 fixture from test_common.h.
 */

#include "test_common.h"

static float bf16_val(nv_bfloat16 v) {
    return __bfloat162float(v);
}

namespace {

// Optional overrides on top of NCCL_EP_GROUP_CONFIG_INIT.
struct GroupOpts {
    ncclEpZeroCopyMode_t zero_copy = NCCL_EP_ZERO_COPY_AUTO;
    ncclEpAlgorithm_t algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
};

ncclEpGroupConfig_t base_group_cfg(const GroupOpts& opts) {
    ncclEpGroupConfig_t cfg = NCCL_EP_GROUP_CONFIG_INIT;
    cfg.algorithm = opts.algorithm;
    cfg.num_experts = kNumExperts;
    cfg.max_dispatch_tokens_per_rank = kNumTokens;
    cfg.max_token_bytes = kHidden * sizeof(nv_bfloat16);
    cfg.rdma_buffer_size = NCCL_EP_AUTO;
    cfg.num_qp_per_rank = NCCL_EP_AUTO;
    cfg.num_channels = NCCL_EP_AUTO;
    cfg.max_recv_tokens_per_rank = static_cast<unsigned int>(kMaxRecvSlots);
    cfg.zero_copy = opts.zero_copy;
    return cfg;
}

// Register `data` as a symmetric window and attach it to `t`.
ncclResult_t attach_symmetric_window(ncclEpTensor_t* t, void* data, size_t bytes, ncclWindow_t* out_win) {
    ncclResult_t r = ncclCommWindowRegister(g_comm, data, bytes, out_win, NCCL_WIN_COLL_SYMMETRIC);
    if (r != ncclSuccess) return r;
    t->win_hdl = *out_win;
    t->win_offset = 0;
    return ncclSuccess;
}

class ZeroCopyTest : public ::testing::Test {
protected:
    int64_t* d_topk_ = nullptr;
    ncclEpTensor_t* topk_ = nullptr;

    void SetUp() override {
        CUDA_ASSERT(cudaMalloc(&d_topk_, kNumTokens * kTopK * sizeof(int64_t)));
        int64_t h[kNumTokens];
        for (int i = 0; i < kNumTokens; ++i) h[i] = expert_for_token(i);
        CUDA_ASSERT(cudaMemcpy(d_topk_, h, sizeof(h), cudaMemcpyHostToDevice));
        NCCL_ASSERT(epTensorCreate(&topk_, 2, ncclInt64, d_topk_, kNumTokens, kTopK));
    }

    void TearDown() override {
        if (topk_) ncclEpTensorDestroy(topk_);
        if (d_topk_) cudaFree(d_topk_);
    }

    // Run one HT FLAT dispatch+combine round on `group`. `windowed_dispatch_out`
    // and `windowed_combine_in` independently force the corresponding tensor to
    // be backed by a freshly-registered symmetric NCCL window. Returns the
    // per-token first-hidden-element value (bf16-rounded) for verification, or
    // an empty vector if `expected_dispatch_err` / `expected_combine_err` were
    // hit (the assertion is reported via the standard gtest macros).
    std::vector<float> run_roundtrip(
        ncclEpGroup_t group,
        bool windowed_dispatch_out,
        bool windowed_combine_in,
        ncclResult_t expected_dispatch_err = ncclSuccess,
        ncclResult_t expected_combine_err = ncclSuccess) {
        // -- Allocate host-side input -----------------------------------------
        std::vector<nv_bfloat16> h_tok(kNumTokens * kHidden);
        for (int i = 0; i < kNumTokens; ++i) {
            float v = static_cast<float>(g_rank * kNumTokens + i + 1);
            for (int hh = 0; hh < kHidden; ++hh) h_tok[i * kHidden + hh] = __float2bfloat16(v);
        }

        // -- Allocate device buffers ------------------------------------------
        // recv buffer (dispatch output / combine input) and combined output use
        // ncclMemAlloc when window-registered so they get a symmetric-friendly
        // allocation; otherwise plain cudaMalloc.
        nv_bfloat16 *d_tok = nullptr, *d_recv = nullptr, *d_out = nullptr;
        float *d_weights = nullptr, *d_recv_w = nullptr;
        int64_t* d_recv_idx = nullptr;
        const size_t recv_bytes = static_cast<size_t>(kMaxRecvSlots) * kHidden * sizeof(nv_bfloat16);

        EXPECT_EQ(cudaMalloc(&d_tok, kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_out, kNumTokens * kHidden * sizeof(nv_bfloat16)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_weights, kNumTokens * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_w, kMaxRecvSlots * kTopK * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&d_recv_idx, kMaxRecvSlots * kTopK * sizeof(int64_t)), cudaSuccess);

        const bool need_window = windowed_dispatch_out || windowed_combine_in;
        if (need_window) {
            EXPECT_EQ(ncclMemAlloc(reinterpret_cast<void**>(&d_recv), recv_bytes), ncclSuccess);
            EXPECT_EQ(cudaMemset(d_recv, 0, recv_bytes), cudaSuccess);
        } else {
            EXPECT_EQ(cudaMalloc(&d_recv, recv_bytes), cudaSuccess);
            EXPECT_EQ(cudaMemset(d_recv, 0, recv_bytes), cudaSuccess);
        }

        std::vector<float> h_w(kNumTokens * kTopK, 1.0f);
        EXPECT_EQ(
            cudaMemcpy(d_tok, h_tok.data(), kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyHostToDevice),
            cudaSuccess);
        EXPECT_EQ(
            cudaMemcpy(d_weights, h_w.data(), kNumTokens * kTopK * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);

        // -- Build tensor descriptors ----------------------------------------
        ncclEpTensor_t *t_tok, *t_recv, *t_out, *t_w, *t_recv_w, *t_recv_idx;
        EXPECT_EQ(epTensorCreate(&t_tok, 2, ncclBfloat16, d_tok, kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(
            epTensorCreate(
                &t_recv,
                2,
                ncclBfloat16,
                windowed_dispatch_out || windowed_combine_in ? nullptr : d_recv,
                kMaxRecvSlots,
                kHidden),
            ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_out, 2, ncclBfloat16, d_out, kNumTokens, kHidden), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_w, 2, ncclFloat32, d_weights, kNumTokens, kTopK), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv_w, 2, ncclFloat32, d_recv_w, kMaxRecvSlots, kTopK), ncclSuccess);
        EXPECT_EQ(epTensorCreate(&t_recv_idx, 2, ncclInt64, d_recv_idx, kMaxRecvSlots, kTopK), ncclSuccess);

        // Window-register the recv buffer; the same window covers dispatch
        // output and combine input (caller picks which path uses it).
        ncclWindow_t recv_win{};
        if (need_window) {
            EXPECT_EQ(attach_symmetric_window(t_recv, d_recv, recv_bytes, &recv_win), ncclSuccess);
        }

        // -- Create handle (collective; ep_group lifetime is per-test) -------
        ncclEpHandle_t h = nullptr;
        EXPECT_EQ(ncclEpCreateHandle(&h, group, NCCL_EP_LAYOUT_FLAT, topk_, nullptr, nullptr, g_stream), ncclSuccess);
        EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);

        // -- Dispatch --------------------------------------------------------
        ncclEpDispatchInputs_t d_in = NCCL_EP_DISPATCH_INPUTS_INIT;
        ncclEpDispatchOutputs_t d_out_s = NCCL_EP_DISPATCH_OUTPUTS_INIT;
        d_in.tokens = t_tok;
        d_in.topk_weights = t_w;
        d_out_s.tokens = t_recv;
        d_out_s.topk_weights = t_recv_w;
        d_out_s.topk_idx = t_recv_idx;

        ncclEpDispatchConfig_t dcfg = NCCL_EP_DISPATCH_CONFIG_INIT;
        ncclResult_t disp_r = ncclEpDispatch(h, &d_in, &d_out_s, nullptr, &dcfg, g_stream);
        EXPECT_EQ(disp_r, expected_dispatch_err);

        std::vector<float> vals;
        if (disp_r == ncclSuccess && expected_combine_err != ncclInvalidArgument) {
            EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
        }

        // -- Combine (only if dispatch succeeded; we keep combine input fresh
        //    from the dispatch output buffer, so the windowed-ness is identical
        //    to windowed_dispatch_out -- windowed_combine_in toggles whether we
        //    *pass* the windowed descriptor to combine).
        if (disp_r == ncclSuccess) {
            ncclEpCombineInputs_t c_in = NCCL_EP_COMBINE_INPUTS_INIT;
            ncclEpCombineOutputs_t c_out_s = NCCL_EP_COMBINE_OUTPUTS_INIT;

            // For ZeroCopyRejectsNonWindowCombine we need a non-window descriptor
            // for the combine input even though dispatch wrote into a windowed
            // buffer. Build a sibling descriptor that points at the same data
            // via .data instead of .win_hdl.
            ncclEpTensor_t* t_combine_in = t_recv;
            ncclEpTensor_t* t_combine_in_alias = nullptr;
            if (need_window && !windowed_combine_in) {
                EXPECT_EQ(
                    epTensorCreate(&t_combine_in_alias, 2, ncclBfloat16, d_recv, kMaxRecvSlots, kHidden),
                    ncclSuccess);
                t_combine_in = t_combine_in_alias;
            }

            c_in.tokens = t_combine_in;
            c_out_s.tokens = t_out;
            ncclResult_t comb_r = ncclEpCombine(h, &c_in, &c_out_s, nullptr, g_stream);
            EXPECT_EQ(comb_r, expected_combine_err);

            if (comb_r == ncclSuccess) {
                EXPECT_EQ(cudaStreamSynchronize(g_stream), cudaSuccess);
                std::vector<nv_bfloat16> h_out(kNumTokens * kHidden);
                EXPECT_EQ(
                    cudaMemcpy(h_out.data(), d_out, kNumTokens * kHidden * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost),
                    cudaSuccess);
                vals.resize(kNumTokens);
                for (int i = 0; i < kNumTokens; ++i) vals[i] = bf16_val(h_out[i * kHidden]);
            }
            if (t_combine_in_alias) ncclEpTensorDestroy(t_combine_in_alias);
        }

        // -- Cleanup ---------------------------------------------------------
        EXPECT_EQ(ncclEpHandleDestroy(h), ncclSuccess);
        if (need_window) {
            EXPECT_EQ(ncclCommWindowDeregister(g_comm, recv_win), ncclSuccess);
        }
        ncclEpTensorDestroy(t_tok);
        ncclEpTensorDestroy(t_recv);
        ncclEpTensorDestroy(t_out);
        ncclEpTensorDestroy(t_w);
        ncclEpTensorDestroy(t_recv_w);
        ncclEpTensorDestroy(t_recv_idx);
        cudaFree(d_tok);
        cudaFree(d_out);
        cudaFree(d_weights);
        cudaFree(d_recv_w);
        cudaFree(d_recv_idx);
        if (need_window) ncclMemFree(d_recv);
        else cudaFree(d_recv);
        return vals;
    }

    static void expect_identity_roundtrip(const std::vector<float>& vals) {
        ASSERT_EQ(vals.size(), static_cast<size_t>(kNumTokens));
        for (int i = 0; i < kNumTokens; ++i) {
            float expected = static_cast<float>(g_rank * kNumTokens + i + 1);
            EXPECT_NEAR(vals[i], expected, 0.5f) << "rank " << g_rank << " token " << i;
        }
    }
};

// -- Tests ---------------------------------------------------------------------

// zero_copy = AUTO (zero-init default) -> library-owned staging.
TEST_F(ZeroCopyTest, DefaultStaging) {
    GroupOpts opts;
    ncclEpGroupConfig_t cfg = base_group_cfg(opts);
    ncclEpGroup_t g = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&g, g_comm, &cfg));

    auto vals = run_roundtrip(g, /*windowed_dispatch_out=*/false, /*windowed_combine_in=*/false);
    expect_identity_roundtrip(vals);

    NCCL_ASSERT(ncclEpGroupDestroy(g));
}

// zero_copy = ON -- dispatch output and combine input are user-registered windows.
TEST_F(ZeroCopyTest, ZeroCopyWindowedRoundtrip) {
    GroupOpts opts;
    opts.zero_copy = NCCL_EP_ZERO_COPY_ON;
    ncclEpGroupConfig_t cfg = base_group_cfg(opts);
    ncclEpGroup_t g = nullptr;
    NCCL_ASSERT(ncclEpCreateGroup(&g, g_comm, &cfg));

    auto vals = run_roundtrip(g, /*windowed_dispatch_out=*/true, /*windowed_combine_in=*/true);
    expect_identity_roundtrip(vals);

    NCCL_ASSERT(ncclEpGroupDestroy(g));
}

} // namespace

// -- main ----------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (!ep_bootstrap(argc, argv, "nccl_ep_zero_copy_uid")) return 0;
    int ret = RUN_ALL_TESTS();
    ep_teardown();
    return ret;
}
