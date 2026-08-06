from ast import parse
import os
import argparse
import torch
import torch.distributed as dist
from pace.sg import SGComm, CommConfig
from utils import init_dist, bench, bench_kineto
import torch.cuda.nvtx as nvtx
from functools import partial
import random
import re

def all_to_all_baseline(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = dist.get_world_size(group)

    if world_size > 1:
        if scatter_dim == 1 and gather_dim == 0:
            s, h, d = x.shape
            x_ = x.reshape(s, world_size, h // world_size, d).contiguous()
            x_ = x_.transpose(0, 1).contiguous()
            output = torch.empty_like(x_) # [t, s, h/t, d]
            dist.all_to_all_single(output, x_, group=group, **kwargs)
            output = output.reshape(s * world_size, h // world_size,
                                    d).contiguous()
        elif scatter_dim == 0 and gather_dim == 1:
            s, h, d = x.shape
            x_ = x.reshape(world_size, s // world_size, h, d).contiguous() #[t, s/t, h, d]
            x_ = x_.transpose(1, 2).contiguous() #[t, h, s/t, d]
            output = torch.empty_like(x_)
            dist.all_to_all_single(output, x_, group=group, **kwargs)
            output = output.reshape(h * world_size, s // world_size, #[t*h, s/t, d]
                                    d).contiguous()
            output = output.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError(
                'Only support scatter_dim=1 and gather_dim=0 or scatter_dim=0 and gather_dim=1.'
            )
        return output
    return x

def all_to_all_only(x, group=None):
    """
    Only perform all-to-all operation.
    """
    world_size = dist.get_world_size(group)

    if world_size > 1:
        output = torch.empty_like(x)
        dist.all_to_all_single(output, x, group=group)
        return output
    return x


# --- Strided-S input layout contract (ulysses CP / USP qkv split) -------------
# Ported from test/test_sg_strided_api.py. CPU-only checks of
# SGComm._validate_sg_input (pure Python — no GPU / torch.distributed / runtime
# needed), plus the end-to-end strided-QKV check in test_main. Both run only
# when --strided is passed. Mirrors the contract documented on
# SGComm.scatter_gather and enforced by sgcomm.py:_validate_sg_input.
def _make_sg_validator():
    """Build an SGComm instance without running __init__ (no group/GPU needed)."""
    obj = SGComm.__new__(SGComm)
    obj._STRIDE_ALIGN = SGComm._STRIDE_ALIGN
    return obj


def _expect_strided_error(v, t, idx, pattern):
    """Assert _validate_sg_input rejects t with a ValueError matching `pattern`."""
    try:
        v._validate_sg_input(t, idx)
    except ValueError as e:
        assert re.search(pattern, str(e)), f'error {e!r} did not match {pattern!r}'
        return
    raise AssertionError(f'expected ValueError matching {pattern!r} for input {idx}')


def _run_strided_validator_checks():
    """CPU-only layout-contract checks (every case from test_sg_strided_api.py)."""
    v = _make_sg_validator()

    # contiguous fast path passes (no checks applied)
    t = torch.zeros((8, 16, 32), dtype=torch.float32)
    assert t.is_contiguous()
    v._validate_sg_input(t, 0)

    # valid strided QKV Q passes (fp32 + bf16); strides (G*H*D, D, 1)
    for S, G, H, D, dtype in [(8, 3, 16, 32, torch.float32), (8, 3, 16, 16, torch.bfloat16)]:
        q = torch.empty((S, G, H, D), dtype=dtype)[:, 0, :, :]
        assert not q.is_contiguous()
        assert q.stride() == (G * H * D, D, 1)
        v._validate_sg_input(q, 0)

    base = torch.zeros((8, 16, 32), dtype=torch.float32)
    # stride(2) != 1 rejected
    _expect_strided_error(v, base[:, :, ::2], 0, 'stride.2.')
    # stride(1) != D rejected
    _expect_strided_error(v, base[:, ::2, :], 0, 'stride.1.')
    # stride(0) < H*D rejected
    H, D = 16, 32
    storage = torch.empty((8, 16, 32), dtype=torch.float32)
    _expect_strided_error(v, torch.as_strided(storage, (4, H, D), (H * D - 1, D, 1)), 0, 'stride.0.')
    # stride(0)*esize not 16-byte aligned rejected
    _expect_strided_error(v, torch.as_strided(storage, (2, H, D), (H * D + 1, D, 1)), 0, 'stride.0..element_size')
    # size(2)*esize not 16-byte aligned rejected (isolated from the other terms)
    H, D = 16, 6
    storage = torch.empty((8, 16, 96), dtype=torch.float32)
    _expect_strided_error(v, torch.as_strided(storage, (2, H, D), (2 * H * D, D, 1)), 0, 'size.2..element_size')
    # non-3-D rejected
    _expect_strided_error(v, torch.zeros((8, 16, 4, 4), dtype=torch.float32)[:, :, :, ::2], 0, '3-D')
    # input index appears in the message
    _expect_strided_error(v, base[:, :, ::2], 2, 'input 2')


def test_main(args: argparse.Namespace, num_local_ranks: int, num_ranks: int, rank: int,
              seed: int, buffer: SGComm, group: dist.ProcessGroup):

    seq_len_list, hidden = args.seq_len, args.hidden

    h = args.nhead
    n = num_ranks
    d = hidden
    num_sms = args.num_sms
    dtype = torch.float32 if args.dtype_size == 4 else torch.bfloat16
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    def check_d1_sg(sg_tensor):
        assert sg_tensor.shape == (s * n, h // n, d)
        sg_view = sg_tensor.view((n, h // n * s, d))
        for i in range(n):
            if torch.sum(sg_view[i, :, :] != i).item() != 0:
                print(f'[rank {rank}]: sg_view[{i}, :, :] check failed')
                torch.save(sg_view, f'sg_d1_{rank}.h5')
                raise RuntimeError('check_d1_sg failed')

    def check_d0_sg(sg_tensor):
        assert sg_tensor.shape == (s, h, d), sg_tensor.shape
        sg_view = sg_tensor
        for i in range(h):
            if torch.sum(sg_view[:, i, :] != i // (h // n)).item() != 0:
                print(f'[rank {rank}]: sg_view[:, {i}, :] check failed')
                torch.save(sg_view, f'sg_d0_{rank}.h5')
                raise RuntimeError('check_d0_sg failed')

    def test_sg(scatter_dim, gather_dim, input_tensor, rand_input, n):
        if scatter_dim == 0:
            input_tensor = input_tensor.view((s * n, h // n, d))
            rand_input = rand_input.view((s * n, h // n, d))
        kwargs = {"scatter_dim": scatter_dim, "gather_dim": gather_dim}
        if not args.no_check:
            check_func = check_d1_sg if scatter_dim == 1 and gather_dim == 0 else check_d0_sg
            rand_answer = all_to_all_baseline(rand_input, scatter_dim, gather_dim, group=group)
            extra_rands = 5 if args.replicas == -1 else args.replicas - 1
            # different size
            batch_rand_input = [rand_input]
            batch_rand_answer = [rand_answer]
            for i in range(extra_rands):
                if gather_dim == 0:
                    rand_i = torch.randn((rand_input.size(0) // num_ranks // (2 ** (1 + i)) * num_ranks, rand_input.size(1), rand_input.size(2)), dtype=rand_input.dtype)
                else:
                    rand_i = torch.randn((rand_input.size(0), rand_input.size(1) // num_ranks // (2 ** (1 + i)) * num_ranks, rand_input.size(2)), dtype=rand_input.dtype)
                rand_o = all_to_all_baseline(rand_i, scatter_dim, gather_dim, group=group)
                batch_rand_input.append(rand_i)
                batch_rand_answer.append(rand_o)
            mismatch_i = batch_rand_input[:-1] + [torch.rand((rand_input.size(0), rand_input.size(1), 2 * rand_input.size(2)))]
            kwargs["tensor"] = input_tensor
            # normal
            sg_tensor, _ = buffer.scatter_gather(**kwargs)
            check_func(sg_tensor=sg_tensor)
            # async finish
            sg_tensor, event = buffer.scatter_gather(async_finish=True, **kwargs)
            event.current_stream_wait()
            check_func(sg_tensor=sg_tensor)
            # previous_event chaining (overlap primitive; previously
            # unexercised for SG). kwargs still carries tensor=input_tensor.
            pe = buffer.capture()
            sg_tensor, _ = buffer.scatter_gather(previous_event=pe, **kwargs)
            check_func(sg_tensor=sg_tensor)
            # random input
            # normal
            kwargs["tensor"] = rand_input
            sg_tensor, _ = buffer.scatter_gather(**kwargs)
            if not torch.allclose(sg_tensor, rand_answer):
                print(f'[rank {rank}]: random input check failed')
                torch.save((sg_tensor, rand_answer), f'rand_sg_{rank}.h5')
                raise RuntimeError('random input check failed')
            # out= (in-place): write into a caller-provided buffer and confirm
            # both correctness and that the returned tensor IS the out tensor.
            sg_out = torch.empty_like(sg_tensor)
            sg_tensor_out, _ = buffer.scatter_gather(out=sg_out, **kwargs)
            assert sg_tensor_out.data_ptr() == sg_out.data_ptr(), f'[rank {rank}]: out= did not write in place'
            assert torch.allclose(sg_out, rand_answer), f'[rank {rank}]: out= random input check failed'
            # async finish
            sg_tensor, event = buffer.scatter_gather(async_finish=True, **kwargs)
            event.current_stream_wait()
            assert torch.allclose(sg_tensor, rand_answer), f'[rank {rank}]: random input check failed'

            # 3 async finish
            async_outs = []
            del kwargs["tensor"]
            for x in batch_rand_input:
                ret = buffer.scatter_gather(tensor=x, async_finish=True, **kwargs)
                async_outs.append(ret)
            for out in async_outs:
                out[1].current_stream_wait()
            for i in range(extra_rands + 1):
                if not torch.allclose(batch_rand_answer[i], async_outs[i][0]):
                    print(f'[rank {rank}]: {scatter_dim=} {gather_dim=} {i}-th async seq test failed, saved to rand_batch_{rank}_{i}.h5')
                    torch.save((batch_rand_answer[i], async_outs[i][0]), f'rand_batch_{rank}_{i}.h5')
            # batch input
            outs, _ = buffer.scatter_gather(batch_rand_input, **kwargs)
            for i in range(len(batch_rand_input)):
                if not torch.allclose(outs[i], batch_rand_answer[i]):
                    save_file = f'b_{rank}.h5'
                    torch.save((outs, batch_rand_answer), save_file)
                    raise Exception(f'[rank {rank}]: batch input check failed, save to {save_file}')


            # batch input
            outs, event = buffer.scatter_gather(batch_rand_input, async_finish=True, **kwargs)
            torch.randn((5120, 5120), dtype=torch.bfloat16) @ torch.randn((5120, 5120), dtype=torch.bfloat16)
            event.current_stream_wait()
            for i in range(len(batch_rand_input)):
                assert torch.allclose(outs[i], batch_rand_answer[i]), f'[rank {rank}]: batch input check failed'

            dist.barrier()
            if rank == 0:
                print(f'[rank {rank}]: sg_alltoall_s{scatter_dim}g{gather_dim} check passed')

        if not args.no_bench:
            # Performance Testing
            dist.barrier()
            with nvtx.range(f'sg_alltoall_s{scatter_dim}g{gather_dim}'):
                t = bench(lambda: buffer.scatter_gather(input_tensor, **kwargs), post_fn=dist.barrier)[0]
                if rank == 0 or True:
                    print(f'[rank {rank}]: sg_alltoall_s{scatter_dim}g{gather_dim} : bandwidth: {total_bytes / t / 1e9:.2f} GBps')
                dist.barrier()

            def test_func():
                buffer.scatter_gather(input_tensor, **kwargs)

            def kineto_bench():
                if num_sms > 0:
                    k2p = {
                        'scattergather_kernel' : 1,
                    }
                else:
                    if num_local_ranks != num_ranks:
                        k2p = {
                            'scattergathercordkernel' : 1,
                        }
                    else:
                        use_slots = buffer.get_comm_slots(input_tensor)
                        k2p = {
                            'Memcpy DtoD (Device -> Device)' : num_ranks * use_slots,
                            'Memcpy PtoP (Device -> Device)' : (num_ranks - 1) * use_slots,
                        }
                def tfunc(bundle):
                    if num_local_ranks != num_ranks or num_sms > 0:
                        send_t = sum(bundle)
                    else:
                        send_t = (sum(bundle[0]) + sum(bundle[1])) / 2 # it is not accurate actually, they are on two different streams
                    recv_t = 0
                    return send_t, recv_t, send_t + recv_t
                with nvtx.range(f'sg_alltoall_s{scatter_dim}g{gather_dim}'):
                    ret = bench_kineto(test_func, kernel_names=tuple(k2p.keys()), barrier_comm_profiling=True, suppress_kineto_output=True, num_kernels_per_period=k2p, trace_path=f'kineto_{rank}_s{scatter_dim}g{gather_dim}.json' if args.save_kineto and seed == 0 else None)
                if all([v is None for v in ret]):
                    print('kineto failed, it is normal if you launch program with nsys')
                else:
                    send_t, recv_t, total_t = tfunc(ret)
                    if rank == 0 or True:
                        print(f'[rank {rank}]: sg_alltoall_s{scatter_dim}g{gather_dim} : total time {total_t * 1e6:.2f} us, bandwidth: {total_bytes / total_t / 1e9:.2f} GBps')

            if not args.no_compare_baseline:
                with nvtx.range(f'nccl_all_to_all_baseline_s{scatter_dim}g{gather_dim}'):
                    t = bench(lambda: all_to_all_baseline(input_tensor, scatter_dim, gather_dim, group=group), post_fn=dist.barrier)[0]
                    if rank == 0 or True:
                        print(f'[rank {rank}]: nccl_all_to_all_s{scatter_dim}g{gather_dim} within {num_ranks} peers, bandwidth: {total_bytes / t / 1e9:.2f} GBps')
                    dist.barrier()

                with nvtx.range(f'nccl_all_to_all_only_s{scatter_dim}g{gather_dim}'):
                    t = bench(lambda: all_to_all_only(input_tensor, group=group), post_fn=dist.barrier)[0]
                    if rank == 0 or True:
                        print(f'[rank {rank}]: nccl_all_to_all_only_s{scatter_dim}g{gather_dim} within {num_ranks} peers, bandwidth: {total_bytes / t / 1e9:.2f} GBps')
        dist.barrier()

    # Strided-S (QKV-view) inputs end-to-end, gated by --strided. Mirrors
    # test_sg_strided_gpu.py sizing so the scatter dim is divisible by
    # num_ranks. The layout contract is checked first via _validate_sg_input,
    # then the strided source is scatter-gathered directly (no copy-in).
    if args.strided and not args.no_check:
        def check_strided(scatter_dim, gather_dim, dtype):
            n = num_ranks
            S, G, H, D = (8 * n, 3, 8, 32) if scatter_dim == 0 else (8, 3, 8 * n, 32)
            storage = (torch.arange(S * G * H * D, dtype=dtype, device='cuda').reshape(S, G, H, D) + rank)
            q = storage[:, 0, :, :]  # (S, H, D), strides (G*H*D, D, 1)
            assert not q.is_contiguous()
            assert q.stride() == (G * H * D, D, 1)
            buffer._validate_sg_input(q, 0)
            ref = all_to_all_baseline(q.contiguous(), scatter_dim, gather_dim, group=group)
            got, _ = buffer.scatter_gather(q, scatter_dim=scatter_dim, gather_dim=gather_dim)
            assert torch.equal(got, ref), f'[rank {rank}]: strided QKV s{scatter_dim}g{gather_dim} mismatch'
            # non-power-of-2 row_bytes: D*esize=96, H*D*esize=768 (not pow2) —
            # exercises the division path in the s0g1 strided read.
            if dtype == torch.float32 and scatter_dim == 0:
                S, G, H, D = 8 * n, 2, 8, 24
                storage = (torch.arange(S * G * H * D, dtype=dtype, device='cuda').reshape(S, G, H, D) + rank)
                q = storage[:, 0, :, :]
                ref = all_to_all_baseline(q.contiguous(), scatter_dim, gather_dim, group=group)
                got, _ = buffer.scatter_gather(q, scatter_dim=scatter_dim, gather_dim=gather_dim)
                assert torch.equal(got, ref), f'[rank {rank}]: strided D=24 s0g1 mismatch'
        for dtype in (torch.float32, torch.bfloat16):
            for scatter_dim, gather_dim in [(0, 1), (1, 0)]:
                check_strided(scatter_dim, gather_dim, dtype)
        dist.barrier()
        if rank == 0:
            print(f'[rank {rank}]: strided QKV check passed', flush=True)

    # Iterate over all sequence lengths
    for seq_len in seq_len_list:
        s = seq_len // num_ranks
        total_bytes = s * h * d * dtype.itemsize
        if rank == 0:
            print(f'\n===== Testing seq_len={seq_len} =====')

        input_tensor = torch.ones((s, h, d), dtype=dtype, device='cuda') * rank
        rand_input = torch.randn((s, h, d), dtype=dtype, device='cuda')
        configs = [(1, 0), (0, 1)]
        for scatter_dim, gather_dim in configs:
            test_sg(scatter_dim, gather_dim, input_tensor, rand_input, n)

        # === Sequence Parallel QKV Multi-Stream Test ===
        # Simulate: 3 model blocks on 3 CUDA streams, each block has Q/K/V 3 tensors
        # Total: 3 blocks × 3 QKV = 9 tensors doing scatter-gather communication
        if not args.no_check:
            num_blocks = 3
            streams = [torch.cuda.Stream() for _ in range(num_blocks)]
            for scatter_dim, gather_dim in [(0, 1), (1, 0)]:
                # Generate reference inputs and answers for each block
                all_inputs = []
                all_answers = []
                for block_idx in range(num_blocks):
                    q = torch.randn((s, h, d), dtype=dtype, device='cuda')
                    k = torch.randn((s, h, d), dtype=dtype, device='cuda')
                    v = torch.randn((s, h, d), dtype=dtype, device='cuda')
                    if scatter_dim == 0:
                        q = q.view((s * n, h // n, d))
                        k = k.view((s * n, h // n, d))
                        v = v.view((s * n, h // n, d))
                    qkv_list = [q, k, v]
                    answer_list = [all_to_all_baseline(t, scatter_dim, gather_dim, group=group) for t in qkv_list]
                    all_inputs.append(qkv_list)
                    all_answers.append(answer_list)
                
                default_stream = torch.cuda.current_stream()
                # Test 2: sync - each block launches QKV list on its own stream, wait immediately
                all_results = []
                for block_idx in range(num_blocks):
                    with torch.cuda.stream(streams[block_idx]):
                        # Ensure input tensors (created on default stream) are ready on this stream
                        # A lightweight PyTorch op triggers automatic cross-stream synchronization
                        torch.cuda.current_stream().wait_stream(default_stream)
                        outs, _ = buffer.scatter_gather(
                            all_inputs[block_idx],
                            scatter_dim=scatter_dim,
                            gather_dim=gather_dim,
                        )
                        all_results.append(outs)
                for stream in streams:
                    default_stream.wait_stream(stream)
                for block_idx in range(num_blocks):
                    for qkv_idx in range(3):
                        if not torch.allclose(all_results[block_idx][qkv_idx], all_answers[block_idx][qkv_idx]):
                            fused = all_results[block_idx][qkv_idx]
                            ref = all_answers[block_idx][qkv_idx]
                            diff = (fused != ref)
                            mc = diff.sum().item()
                            idx = torch.nonzero(diff, as_tuple=True)
                            first_n = min(10, mc)
                            fv = fused[idx][:first_n].float()
                            rv = ref[idx][:first_n].float()
                            print(f'[rank {rank}]: seq_parallel_qkv sync check failed at block {block_idx}, qkv {qkv_idx}, s{scatter_dim}g{gather_dim} '
                                    f'MISMATCH: count={mc}/{fused.numel()} shape={tuple(fused.shape)} '
                                    f'first_idx={tuple(dim[:first_n].tolist() for dim in idx)} '
                                    f'fused_vals={fv.tolist()} ref_vals={rv.tolist()}')
                            torch.save({'fused': fused, 'ref': ref, 'input': all_inputs[block_idx][qkv_idx],
                                        'all_fused': all_results[block_idx], 'all_ref': all_answers[block_idx],
                                        'all_input': all_inputs[block_idx]},
                                        f'seq_par_sync_{rank}_b{block_idx}_q{qkv_idx}_s{scatter_dim}g{gather_dim}.h5')
                            raise RuntimeError('seq_parallel QKV sync check failed')

                # Test 1: async finish - each block launches QKV list on its own stream
                all_results = []
                all_events = []
                for block_idx in range(num_blocks):
                    with torch.cuda.stream(streams[block_idx]):
                        # Ensure input tensors (created on default stream) are ready on this stream
                        # A lightweight PyTorch op triggers automatic cross-stream synchronization
                        torch.cuda.current_stream().wait_stream(default_stream)
                        outs, event = buffer.scatter_gather(
                            all_inputs[block_idx],
                            scatter_dim=scatter_dim,
                            gather_dim=gather_dim,
                            async_finish=True,
                        )
                        all_results.append(outs)
                        all_events.append(event)

                # Wait for all events on default stream and verify
                for block_idx in range(num_blocks):
                    all_events[block_idx].current_stream_wait()
                for block_idx in range(num_blocks):
                    for qkv_idx in range(3):
                        if not torch.allclose(all_results[block_idx][qkv_idx], all_answers[block_idx][qkv_idx]):
                            fused = all_results[block_idx][qkv_idx]
                            ref = all_answers[block_idx][qkv_idx]
                            diff = (fused != ref)
                            mc = diff.sum().item()
                            idx = torch.nonzero(diff, as_tuple=True)
                            first_n = min(10, mc)
                            fv = fused[idx][:first_n].float()
                            rv = ref[idx][:first_n].float()
                            print(f'[rank {rank}]: seq_parallel_qkv async check failed at block {block_idx}, qkv {qkv_idx}, s{scatter_dim}g{gather_dim} '
                                    f'MISMATCH: count={mc}/{fused.numel()} shape={tuple(fused.shape)} '
                                    f'first_idx={tuple(dim[:first_n].tolist() for dim in idx)} '
                                    f'fused_vals={fv.tolist()} ref_vals={rv.tolist()}')
                            torch.save({'fused': fused, 'ref': ref, 'input': all_inputs[block_idx][qkv_idx],
                                        'all_fused': all_results[block_idx], 'all_ref': all_answers[block_idx],
                                        'all_input': all_inputs[block_idx]},
                                        f'seq_par_async_{rank}_b{block_idx}_q{qkv_idx}_s{scatter_dim}g{gather_dim}.h5')
                            raise RuntimeError('seq_parallel QKV async check failed')

                dist.barrier()
                if rank == 0:
                    print(f'[rank {rank}]: seq_parallel_qkv_s{scatter_dim}g{gather_dim} check passed (3 blocks × 3 QKV = 9 tensors)')

def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks, 'gloo' if args.no_compare_baseline else 'nccl')

    use_sep_stdout = args.sep_out_pfx is not None
    if use_sep_stdout:
        import sys
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)
        newout = open(f'{args.sep_out_pfx}_{rank}.txt', 'w')
        os.dup2(newout.fileno(), original_stdout_fd)
        os.dup2(newout.fileno(), original_stderr_fd)
        print(f'[rank {rank}]: use seperate stdout/stderr', flush=True)

    assert num_local_ranks % args.split_intra == 0
    num_local_ranks = num_local_ranks // args.split_intra
    # max_data_bytes multiplied by 2 is for batch_input, batch_input will use 1 + 1/2 + 1/4 + ... < 2 times the single input size
    config = CommConfig(slot_unroll=args.unroll, nvl_ring_size=args.nvl_ring, rdma_ring_size=args.rdma_ring, num_sms=args.num_sms)
    buffer = SGComm(group, num_local_ranks=num_local_ranks, config=config)
    # is_ring_mode() is the reform-fixed query (was IsRingMode vs bound
    # IsDataRingMode name mismatch); exercise it so a regression is caught.
    assert isinstance(buffer.is_ring_mode(), bool), 'is_ring_mode must return a bool'
    if rank == 0:
        print(f'[rank {rank}]: is_ring_mode={buffer.is_ring_mode()}')
        # Shared memory footprint: the ring buffer + signal buffer, allocated
        # once at connect (nothing is allocated after that).
        sm = buffer.runtime.get_shared_memory_bytes()
        print(f'[rank {rank}]: SG shared memory = {sm/1e6:.2f} MB (buffer + {4*1024*1024/1e6:.1f} MB signal)', flush=True)
    for seed in range(args.repeat):
        test_main(args, num_local_ranks, num_ranks, rank, seed, buffer, group)


    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f'[rank {rank}]: test OK', flush=True)
    if use_sep_stdout:
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        newout.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test intranode SGComm')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of processes to spawn (default: torch.cuda.device_count())')
    parser.add_argument('--seq-len', type=int, nargs='+', default=[65536],
                       help='Total sequence length(s) to test (default: [65536])')
    parser.add_argument('--hidden', type=int, default=128,
                       help='Hidden dimension size (default: 128)')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument("--sep-out-pfx", type=str, default=None, help="split output file prefix for each rank, stored as <prefix>_<rank>.txt, default: None, means mixed stdout")
    parser.add_argument("--no-check", action='store_true', help='skip correction test')
    parser.add_argument("--no-bench", action='store_true', help='skip benchmark')
    parser.add_argument("--repeat", type=int, default=1, help='number of benchmark repeats')
    parser.add_argument("--dtype-size", type=int, default=2, choices=[2, 4], help='data type size in bytes, 4 : int32, 2: bfloat16. (default: 2)')
    parser.add_argument("--num-sms", type=int, default=0, help='number of SMs to use for SGComm kernels (default: 32)')
    parser.add_argument("--nvl-ring", type=int, default=4, help='nvl ring size')
    parser.add_argument("--rdma-ring", type=int, default=4, help='rdma ring size')
    parser.add_argument("--unroll", type=int, default=32, help='segment length, unrolls of 1024 float4')
    parser.add_argument("--split-intra", type=int, default=1, help='split 1 node into required micro nodes, for simulation use')
    parser.add_argument("--no-compare-baseline", action='store_true', help='do not compare with baseline')
    parser.add_argument("--save-kineto", action='store_true', help='store kineto trace for profiling')
    parser.add_argument("--replicas", type=int, default=-1, help='batch-size knob: number of extra batch tensors to exercise (-1 = default 5)')
    parser.add_argument("--lid", type=int, default=0, help='local id for direct launch')
    parser.add_argument("--nlid", type=int, default=0, help='number of local processes for direct launch')
    parser.add_argument('--strided', action='store_true',
                       help='also exercise strided-S (QKV-view) inputs: run the '
                            'layout-contract validator checks and strided '
                            'end-to-end scatter-gather')
    args = parser.parse_args()

    if args.strided:
        # CPU-only layout-contract checks; runs once on the launcher, before the
        # distributed spawn (no GPU/distributed runtime required).
        _run_strided_validator_checks()
        print('strided validator checks passed', flush=True)

    if args.nlid > 0:
        test_loop(args.lid, args.nlid, args)
    else:
        num_processes = args.num_processes
        if num_processes is None:
            import torch
            num_processes = torch.cuda.device_count()
        torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
