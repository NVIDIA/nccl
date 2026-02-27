/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_ep.h>

// Custom allocator wrappers (can be replaced with custom memory pool)
static cudaError_t torchMalloc(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

static cudaError_t torchFree(void* ptr) {
    return cudaFree(ptr);
}

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed: NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

// Convert float to bfloat16 (as uint16_t)
static inline uint16_t float_to_bf16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    // Round to nearest even
    uint32_t rounding_bias = 0x00007FFF + ((x >> 16) & 1);
    return static_cast<uint16_t>((x + rounding_bias) >> 16);
}

static void getHostName(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}


void printUsage(const char* programName, int myRank) {
    if (myRank == 0) {
        printf("Usage: %s [OPTIONS]\n", programName);
        printf("Options:\n");
        printf("  -a <ll|ht>                   Set algorithm mode (default: ll)\n");
        printf("                               ll:  Low latency mode (NCCL_EP_ALGO_LOW_LATENCY)\n");
        printf("                               ht:  High throughput mode (NCCL_EP_ALGO_HIGH_THROUGHPUT)\n");
        printf("  -m                           Disable max_tokens_per_rank (only supported with HT mode)\n");
        printf("  -s <none|dispatch|combine|both>  Set send_only mode (default: none)\n");
        printf("                               none:     send_only=false for both dispatch and combine\n");
        printf("                               dispatch: send_only=true for dispatch only\n");
        printf("                               combine:  send_only=true for combine only\n");
        printf("                               both:     send_only=true for both dispatch and combine\n");
        printf("  -c                           Enable cached mode (default: disabled, only supported for HT)\n");
        printf("  -r                           Enable random mode (random topk_idx, skip correctness checks)\n");
        printf("  -t <num>                     Set number of tokens (default: 50)\n");
        printf("  -d <num>                     Set hidden dimension size (default: 7168)\n");
        printf("  -h                           Show this help message\n");
    }
}

int main(int argc, char* argv[])
{
  int myRank, nRanks, localRank = 0;
  ncclEpAlgorithm_t algorithm = NCCL_EP_ALGO_LOW_LATENCY; // Default to 'll' (low latency)
  bool disable_max_tokens = false; // Flag to disable max_tokens_per_rank
  bool dispatch_send_only = false; // send_only flag for dispatch
  bool combine_send_only = false;  // send_only flag for combine
  bool cached_mode = false;        // cached mode flag
  bool random_mode = false;        // random mode flag (random topk_idx, skip checks)
  unsigned int num_tokens = 50; // number of tokens (default)
  unsigned int hidden = 7168;      // hidden dimension size (default)

  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  if (nRanks != 2 && nRanks != 4 && nRanks % 8 != 0) {
    printf("Error: nRanks must be 2, 4 or multiple of 8 for this test\n");
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // Parse command line arguments using getopt
  int opt;
  while ((opt = getopt(argc, argv, "a:ms:crt:d:h")) != -1) {
    switch (opt) {
      case 'a':
        if (strcmp(optarg, "ll") == 0) {
          algorithm = NCCL_EP_ALGO_LOW_LATENCY;
        } else if (strcmp(optarg, "ht") == 0) {
          algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
        } else {
          if (myRank == 0) {
            printf("Error: Invalid algorithm '%s'. Use 'll' or 'ht'\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'm':
        disable_max_tokens = true;
        break;
      case 's':
        if (strcmp(optarg, "none") == 0) {
          dispatch_send_only = false;
          combine_send_only = false;
        } else if (strcmp(optarg, "dispatch") == 0) {
          dispatch_send_only = true;
          combine_send_only = false;
        } else if (strcmp(optarg, "combine") == 0) {
          dispatch_send_only = false;
          combine_send_only = true;
        } else if (strcmp(optarg, "both") == 0) {
          dispatch_send_only = true;
          combine_send_only = true;
        } else {
          if (myRank == 0) {
            printf("Error: Invalid send_only mode '%s'. Use 'none', 'dispatch', 'combine', or 'both'\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'c':
        cached_mode = true;
        break;
      case 'r':
        random_mode = true;
        break;
      case 't':
        num_tokens = static_cast<unsigned int>(atoi(optarg));
        if (num_tokens == 0) {
          if (myRank == 0) {
            printf("Error: Invalid num_tokens '%s'. Must be a positive integer.\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'd':
        hidden = static_cast<unsigned int>(atoi(optarg));
        if (hidden == 0) {
          if (myRank == 0) {
            printf("Error: Invalid hidden size '%s'. Must be a positive integer.\n", optarg);
            printUsage(argv[0], myRank);
          }
          MPI_Finalize();
          exit(EXIT_FAILURE);
        }
        break;
      case 'h':
        printUsage(argv[0], myRank);
        MPI_Finalize();
        exit(EXIT_SUCCESS);
      case '?':
        if (myRank == 0) {
          printUsage(argv[0], myRank);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
      default:
        if (myRank == 0) {
          printUsage(argv[0], myRank);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
  }

  // -m (NCCL_EP_AUTO for max_tokens_per_rank) is intended for HT mode only.
  // Not yet supported in the current release; code paths are kept for future use.
  if (disable_max_tokens) {
    if (myRank == 0) {
      if (algorithm != NCCL_EP_ALGO_HIGH_THROUGHPUT)
        printf("Error: -m is only applicable to HT mode (-a ht)\n");
      else
        printf("Error: -m (NCCL_EP_AUTO for max_tokens_per_rank) is not yet supported.\n"
               "       This feature will be available in a future release for HT mode.\n");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];

  const unsigned int ELEMENTS_TESTED_PER_TOKEN = 10;
  unsigned int top_k = std::min(8, nRanks); // DeepSeek v3 has 8, how many experts each token goes to
  unsigned int num_experts = std::min(256u, top_k * nRanks); // DeepSeek v3
  unsigned int num_local_experts = num_experts / nRanks;
  unsigned int local_experts_start = num_local_experts * myRank;
  unsigned int local_experts_end = local_experts_start + num_local_experts; // exclusive

  if (num_experts % nRanks != 0) {
    if (myRank == 0) printf("Error: num_experts (%u) must be divisible by nRanks (%d)\n", num_experts, nRanks);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
  if (top_k > num_local_experts) {
    if (myRank == 0) printf("Error: top_k (%u) must be less than or equal to num_local_experts (%u)\n", top_k, num_local_experts);
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t s;

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast(static_cast<void*>(&id), sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaStreamCreate(&s));

  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  ncclEpGroup_t ep_group;
  ncclEpGroupConfig_t config;
  config.version = 1;                                    // Structure version
  config.algorithm = algorithm;                          // Algorithm type (set by command line)
  config.num_experts = num_experts;
  // max_tokens_per_rank is the per-rank batch size (max tokens any single rank will send).
  config.max_tokens_per_rank = disable_max_tokens ? NCCL_EP_AUTO : num_tokens;
  config.token_size_bytes = hidden * 2;                  // bfloat16
  config.rdma_buffer_size = NCCL_EP_AUTO;               // NCCL_EP_AUTO for auto configuration, internally uses the hint
  config.num_qp_per_rank = NCCL_EP_AUTO;                // Default is 24, see internode_ll.cu:181 for the minimum
  config.num_channels = NCCL_EP_AUTO;                   // Number of communication channels

  const char* algorithm_name = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? "LOW_LATENCY" : "HIGH_THROUGHPUT";
  printf("Rank %d: Testing ncclEpCreateGroup with algorithm: %s%s\n", myRank, algorithm_name,
         disable_max_tokens ? " (no max_tokens_per_rank)" : "");
  NCCLCHECK(ncclEpCreateGroup(&ep_group, comm, &config, s, torchMalloc, torchFree));

  ncclNDTensor_t topk_idx;
  NCCLCHECK(ncclEpTensorCreate(ep_group, &topk_idx, 2, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(top_k)));
  int64_t *topk_idx_host = new int64_t[num_tokens * top_k];

  if (random_mode) {
    // Random mode: first expert is random, rest are deterministic to avoid repetitions
    srand(myRank + 42);  // Seed with rank for reproducibility
    for (int i = 0; i < num_tokens; i++) {
      // First expert is random
      int64_t first_expert = rand() % num_experts;
      topk_idx_host[i * top_k + 0] = first_expert;

      // Remaining experts are deterministic based on first choice (no repetitions)
      for (int j = 1; j < top_k; j++) {
        topk_idx_host[i * top_k + j] = (first_expert + j) % num_experts;
      }
    }
    if (myRank == 0) {
      printf("Random mode enabled: first expert random, rest deterministic (no repetitions)\n");
    }
  } else {
    // Deterministic mode: send each token to top_k number of semi-random experts, equal distribution
    for (int i = 0; i < num_tokens; i++) {
      for (int j = 0; j < top_k; j++) {
        topk_idx_host[i * top_k + j] = (local_experts_end + j) % num_experts;
      }
    }
  }
  CUDACHECK(cudaMemcpy(topk_idx.data, topk_idx_host, num_tokens * top_k * sizeof(int64_t), cudaMemcpyHostToDevice));

  // Create recv_expert_counter host tensor for ncclEpCreateHandle (only when disable_max_tokens is true)
  ncclNDTensor_t* handle_local_tensors[1] = {nullptr};
  unsigned int handle_num_local_tensors = 0;
  ncclNDTensor_t handle_recv_expert_counter;
  if (disable_max_tokens) {
    handle_recv_expert_counter.ndim = 1;
    handle_recv_expert_counter.datatype = ncclInt32;
    handle_recv_expert_counter.strides = new unsigned int[1];
    handle_recv_expert_counter.strides[0] = 1;
    handle_recv_expert_counter.tag = NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST;
    handle_recv_expert_counter.flags = NCCL_EP_TENSOR_FLAG_NONE;
    handle_recv_expert_counter.sizes = new unsigned int[1];
    handle_recv_expert_counter.sizes[0] = num_local_experts;
    CUDACHECK(cudaHostAlloc(&handle_recv_expert_counter.data, num_local_experts * sizeof(int), cudaHostAllocMapped));
    handle_local_tensors[0] = &handle_recv_expert_counter;
    handle_num_local_tensors = 1;
  }

  printf("Rank %d: Testing ncclEpCreateHandle\n", myRank);
  ncclEpHandle_t ep_handle;
  NCCLCHECK(ncclEpCreateHandle(&ep_handle, ep_group, &topk_idx, handle_local_tensors, handle_num_local_tensors, nullptr, s));
  CUDACHECK(cudaStreamSynchronize(s));

  // max_tokens_per_rank is the per-rank dispatch count.
  // num_recv_tokens is the max tokens this rank can receive (nRanks * max_tokens_per_rank).
  unsigned int num_recv_tokens = config.max_tokens_per_rank * nRanks;
  if (disable_max_tokens) {
    NCCLCHECK(ncclEpHandleGetNumRecvTokens(ep_handle, &num_recv_tokens));
  }

  ncclEpDispatchConfig_t dispatch_config;
  dispatch_config.round_scales = 0; // Not testing this parameter atm

  // Array sizes depend on algorithm: LL uses 1 input/output, HT use 3 inputs/3 outputs
  int num_inputs = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? 1 : 3;
  int num_outputs = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? 1 : 3;
  int num_local_tensors = (algorithm == NCCL_EP_ALGO_LOW_LATENCY) ? 1 : 0;

  ncclNDTensor_t *inputs[3]; // Max size for HT
  inputs[0] = new ncclNDTensor_t;

  ncclNDTensor_t *outputs[3]; // Max size for HT
  outputs[0] = new ncclNDTensor_t;
  if (algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    outputs[1] = new ncclNDTensor_t;
    outputs[2] = new ncclNDTensor_t;
  }

  ncclNDTensor_t *local_tensors[1];
  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    local_tensors[0] = new ncclNDTensor_t;
  }

  NCCLCHECK(ncclEpTensorCreate(ep_group, inputs[0], 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(hidden)));
  ncclNDTensor_t topk_weights;
  if (algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    // HT: topk_weights and topk_idx are dispatch inputs
    NCCLCHECK(ncclEpTensorCreate(ep_group, &topk_weights, 2, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(top_k)));
    inputs[1] = &topk_weights;
    topk_idx.tag = NCCL_EP_TENSOR_TAG_TOPK_IDX;
    inputs[2] = &topk_idx;
  } else {
    NCCLCHECK(ncclEpTensorCreate(ep_group, &topk_weights, 2, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(top_k)));
  }
  // outputs[0] shape depends on algorithm
  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    // LL mode: outputs[0] is [num_local_experts, num_recv_tokens, hidden]
    NCCLCHECK(ncclEpTensorCreate(ep_group, outputs[0], 3, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, static_cast<unsigned int>(num_local_experts), num_recv_tokens, static_cast<unsigned int>(hidden)));
  } else {
    // HT mode: outputs[0] is [num_tokens, hidden]
    NCCLCHECK(ncclEpTensorCreate(ep_group, outputs[0], 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(hidden)));
  }

  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    NCCLCHECK(ncclEpTensorCreate(ep_group, local_tensors[0], 1, ncclInt32, NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE, static_cast<unsigned int>(num_local_experts)));
  }

  // In HT mode, outputs[1] (recv_topk_weights) and outputs[2] (recv_topk_idx) are 2D tensors [num_recv_tokens, top_k]
  if (algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    NCCLCHECK(ncclEpTensorCreate(ep_group, outputs[1], 2, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS, static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(top_k)));
    NCCLCHECK(ncclEpTensorCreate(ep_group, outputs[2], 2, ncclInt64, NCCL_EP_TENSOR_TAG_TOPK_IDX, static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(top_k)));
  }

  // Fill the first ELEMENTS_TESTED_PER_TOKEN elements of each token with a special value based on the current rank
  uint16_t *input_host = new uint16_t[num_tokens * hidden]();
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
      input_host[i * hidden + j] = static_cast<uint16_t>(0x1000 + myRank);
    }
  }
  CUDACHECK(cudaMemcpy(inputs[0]->data, input_host, num_tokens * hidden * 2, cudaMemcpyHostToDevice));
  delete[] input_host;

  // Create topk_weights (used as input for HT mode, local tensor for LL mode)

  float *topk_weights_host = new float[num_tokens * top_k];
  for (int i = 0; i < num_tokens; ++i) {
    for (int j = 0; j < top_k; ++j) {
      topk_weights_host[i * top_k + j] = 1.0f / top_k; // Equal weights
    }
  }
  CUDACHECK(cudaMemcpy(topk_weights.data, topk_weights_host, num_tokens * top_k * sizeof(float), cudaMemcpyHostToDevice));
  delete[] topk_weights_host;

  // Host buffer for first phase dispatch output (kept for cached mode comparison in HT mode)
  uint16_t *first_dispatch_output0_host = nullptr;

  printf("Rank %d: Testing ncclEpDispatch (send_only=%s)\n", myRank, dispatch_send_only ? "true" : "false");
  NCCLCHECK(ncclEpDispatch(ep_handle, inputs, num_inputs, outputs, num_outputs,
    local_tensors, num_local_tensors,
    dispatch_send_only /* send_only */, &dispatch_config, s));

  printf("Rank %d: Testing ncclEpComplete\n", myRank);
  NCCLCHECK(ncclEpComplete(ep_handle, nullptr, s));
  CUDACHECK(cudaStreamSynchronize(s));
  // Read recv_count tensor to use for validation
  // LL mode: allocated and copied from device local_tensors[0]
  // HT mode with disable_max_tokens: points to handle_local_tensors[0] (already host memory)
  // HT mode without disable_max_tokens: nullptr (no validation available)
  int *recv_count_host = nullptr;
  bool should_free_recv_count = false;
  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    recv_count_host = new int[num_local_experts];
    CUDACHECK(cudaMemcpy(recv_count_host, local_tensors[0]->data, num_local_experts * sizeof(int), cudaMemcpyDeviceToHost));
    should_free_recv_count = true;
  } else if (disable_max_tokens && handle_local_tensors[0] != nullptr) {
    recv_count_host = static_cast<int*>(handle_local_tensors[0]->data);
  }

  unsigned int recv_from_expert_start = (local_experts_start + num_experts - num_local_experts) % num_experts;
  unsigned int recv_rank = recv_from_expert_start / num_local_experts;

  // Check the first ELEMENTS_TESTED_PER_TOKEN elements of each token
  // For LL mode: use recv_count to validate per-expert token counts
  // For HT mode: use the simple approach
  bool dispatch_check_passed = true;

  if (!random_mode && algorithm == NCCL_EP_ALGO_LOW_LATENCY && recv_count_host != nullptr) {
    // LL mode: outputs[0] is [num_local_experts, num_recv_tokens, hidden]
    uint16_t *output_host = new uint16_t[num_local_experts * num_recv_tokens * hidden]();
    CUDACHECK(cudaMemcpy(output_host, outputs[0]->data,
                         num_local_experts * num_recv_tokens * hidden * 2,
                         cudaMemcpyDeviceToHost));

    for (unsigned int e = 0; e < num_local_experts; e++) {
      int expected_count = num_tokens; // Each expert receives num_tokens tokens (one from each rank)
      if (recv_count_host[e] != expected_count) {
        printf("Recv_count check failed! Rank %d, expert %d: expected %d, got %d\n",
               myRank, e, expected_count, recv_count_host[e]);
        dispatch_check_passed = false;
        break;
      }

      // Verify the first recv_count_host[e] tokens for this expert
      for (int t = 0; t < recv_count_host[e] && t < static_cast<int>(num_recv_tokens); t++) {
        size_t token_offset = (e * num_recv_tokens + t) * hidden;
        for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
          uint16_t expected = static_cast<uint16_t>(0x1000 + recv_rank);
          uint16_t actual = output_host[token_offset + j];
          if (actual != expected) {
            printf("Dispatch data check failed! Rank %d, expert %d, token %d, element %d: expected %d, got %d\n",
                   myRank, e, t, j, expected, actual);
            dispatch_check_passed = false;
            break;
          }
        }
        if (!dispatch_check_passed) break;
      }
      if (!dispatch_check_passed) break;
    }
    delete[] output_host;
  } else if (!random_mode) {
    // HT mode or fallback (only check if not in random mode)
    // Check recv_count (only if available)
    if (recv_count_host != nullptr) {
      for (unsigned int e = 0; e < num_local_experts; e++) {
        int expected_count = num_tokens; // Each expert receives num_tokens tokens (one from each rank)
        if (recv_count_host[e] != expected_count) {
          printf("Recv_count check failed! Rank %d, expert %d: expected %d, got %d\n",
                  myRank, e, expected_count, recv_count_host[e]);
          dispatch_check_passed = false;
          break;
        }
      }
    }

    uint16_t *output_host = new uint16_t[num_recv_tokens * hidden]();
    CUDACHECK(cudaMemcpy(output_host, outputs[0]->data, num_recv_tokens * hidden * 2, cudaMemcpyDeviceToHost));
    int expected_count = disable_max_tokens ? num_recv_tokens : num_tokens;
    for (int i = 0; i < expected_count; ++i) {
      for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
        uint16_t expected = static_cast<uint16_t>(0x1000 + recv_rank);
        uint16_t actual = output_host[i * hidden + j];
        if (actual != expected) {
          printf("Dispatch check failed! Rank %d, token %d, element %d: expected %d, got %d\n",
                 myRank, i, j, expected, actual);
          dispatch_check_passed = false;
          break;
        }
      }
      if (!dispatch_check_passed) break;
    }
    // Keep output_host for cached mode comparison, delete later
    first_dispatch_output0_host = output_host;
  } else {
    // Random mode: skip checks, just clean up
  }

  // Consolidated cleanup for recv_count_host
  if (should_free_recv_count && recv_count_host != nullptr) {
    delete[] recv_count_host;
  }

  // In HT mode, copy and verify outputs[1] (recv_topk_weights) and outputs[2] (recv_topk_idx)
  if (!random_mode && algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    // Copy recv_topk_weights (outputs[1]) - [num_recv_tokens, top_k] float32
    float *recv_topk_weights_host = new float[num_recv_tokens * top_k]();
    CUDACHECK(cudaMemcpy(recv_topk_weights_host, outputs[1]->data, num_recv_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy recv_topk_idx (outputs[2]) - [num_recv_tokens, top_k] int64
    int64_t *recv_topk_idx_host = new int64_t[num_recv_tokens * top_k]();
    CUDACHECK(cudaMemcpy(recv_topk_idx_host, outputs[2]->data, num_recv_tokens * top_k * sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Verify recv_topk_weights and recv_topk_idx
    bool ht_outputs_valid = true;
    printf("Rank %d: Verifying recv_topk_weights and recv_topk_idx\n", myRank);

    // Test ELEMENTS_TESTED_PER_TOKEN (or top_k if smaller) per token
    float expected_weight = 1.0f / top_k;
    int weight_errors = 0;
    int idx_errors = 0;

    for (int i = 0; i < num_tokens; i++) {
      for (int j = 0; j < top_k; j++) {
        int offset = i * top_k + j;

        // Check that recv_topk_weights equals 1.0f / top_k
        if (recv_topk_weights_host[offset] != expected_weight) {
          if (weight_errors < 5) {  // Limit error output
            printf("Rank %d: recv_topk_weights[%d][%d] = %f, expected %f\n",
                   myRank, i, j, recv_topk_weights_host[offset], expected_weight);
          }
          weight_errors++;
          ht_outputs_valid = false;
        }

        // Check that recv_topk_idx is within valid range [0, num_experts)
        int64_t idx_val = recv_topk_idx_host[offset];
        if (idx_val < 0 || idx_val >= static_cast<int64_t>(num_experts)) {
          if (idx_errors < 5) {  // Limit error output
            printf("Rank %d: recv_topk_idx[%d][%d] = %ld, expected range [0, %u)\n",
                   myRank, i, j, static_cast<long>(idx_val), num_experts);
          }
          idx_errors++;
          ht_outputs_valid = false;
        }
      }
    }

    if (weight_errors > 0) {
      printf("Rank %d: recv_topk_weights verification failed with %d errors\n", myRank, weight_errors);
    }
    if (idx_errors > 0) {
      printf("Rank %d: recv_topk_idx verification failed with %d errors\n", myRank, idx_errors);
    }

    if (ht_outputs_valid) {
      printf("Rank %d: %s mode recv_topk_weights and recv_topk_idx verification passed\n", myRank, algorithm_name);
    } else {
      dispatch_check_passed = false;
    }

    delete[] recv_topk_weights_host;
    delete[] recv_topk_idx_host;
  }

  if (random_mode) {
    printf("Rank %d: %s Dispatch flow completed (random mode, checks skipped)\n", myRank, algorithm_name);
  } else if (dispatch_check_passed) {
    printf("Rank %d: %s Dispatch flow passed successfully\n", myRank, algorithm_name);
  } else {
    printf("Rank %d: Exiting test due to dispatch failure\n", myRank);
    exit(EXIT_FAILURE);
  }

  printf("Rank %d: Testing %s Combine flow\n", myRank, algorithm_name);

  // Create tensors for combine
  ncclNDTensor_t expert_outputs;

  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
      // LL mode: expert_outputs is 3D [num_local_experts, num_recv_tokens, hidden]
    NCCLCHECK(ncclEpTensorCreate(ep_group, &expert_outputs, 3, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, static_cast<unsigned int>(num_local_experts), static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(hidden)));

    // Fill expert outputs with multiple test values per token
    // The ith element will be (i+1)*2 for all experts
    uint16_t *expert_outputs_host = new uint16_t[config.max_tokens_per_rank * hidden]();

    for (unsigned int t = 0; t < config.max_tokens_per_rank; ++t) {
        for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
            // Set the ith element to (i+1)*2, stored as bf16
            expert_outputs_host[t * hidden + j] = float_to_bf16(static_cast<float>((j + 1) * 2));
        }
    }

    // Copy the same buffer to each expert's slice
    for (int e = 0; e < num_local_experts; ++e) {
      size_t offset_bytes = static_cast<size_t>(e) * num_recv_tokens * hidden * 2;
      CUDACHECK(cudaMemcpy(static_cast<uint8_t*>(expert_outputs.data) + offset_bytes,
                           expert_outputs_host,
                           config.max_tokens_per_rank * hidden * 2,
                           cudaMemcpyHostToDevice));
    }

    delete[] expert_outputs_host;
  } else {
    // HT mode: expert_outputs is 2D [num_recv_tokens, hidden]
    NCCLCHECK(ncclEpTensorCreate(ep_group, &expert_outputs, 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, num_recv_tokens, static_cast<unsigned int>(hidden)));

    // Fill expert outputs with test values
    // The ith element will be (i+1)*2
    uint16_t *expert_outputs_host = new uint16_t[num_recv_tokens * hidden]();
    for (unsigned int t = 0; t < num_recv_tokens; ++t) {
      for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
        expert_outputs_host[t * hidden + j] = float_to_bf16(static_cast<float>((j + 1) * 2));
      }
    }
    CUDACHECK(cudaMemcpy(expert_outputs.data, expert_outputs_host, num_recv_tokens * hidden * 2, cudaMemcpyHostToDevice));
    delete[] expert_outputs_host;
  }

  // Create combined output tensor
  ncclNDTensor_t combined_output;
  NCCLCHECK(ncclEpTensorCreate(ep_group, &combined_output, 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS, static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(hidden)));

  // Setup combine inputs and outputs
  ncclNDTensor_t *combine_inputs[1];
  combine_inputs[0] = &expert_outputs;

  ncclNDTensor_t *combine_outputs[1];
  combine_outputs[0] = &combined_output;

  ncclNDTensor_t *combine_local_tensors[1];

  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    combine_local_tensors[0] = &topk_weights;
  }

  printf("Rank %d: Testing ncclEpCombine (send_only=%s)\n", myRank, combine_send_only ? "true" : "false");
  NCCLCHECK(ncclEpCombine(ep_handle, combine_inputs, 1, combine_outputs, 1,
    combine_local_tensors, algorithm == NCCL_EP_ALGO_LOW_LATENCY ? 1 : 0,
    combine_send_only /* send_only */, nullptr /* config reserved */, s));

  NCCLCHECK(ncclEpComplete(ep_handle, nullptr, s));
  CUDACHECK(cudaStreamSynchronize(s));

  // Verify combine output - check ELEMENTS_TESTED_PER_TOKEN elements per token
  int combine_errors = 0;
  if (!random_mode) {
    uint16_t *combined_output_host = new uint16_t[num_tokens * hidden]();
    CUDACHECK(cudaMemcpy(combined_output_host, combined_output.data, num_tokens * hidden * 2, cudaMemcpyDeviceToHost));

    // Since all experts output the same values and we use equal weights,
    // each token should have the same values as the expert outputs: (i+1)*2
    for (int i = 0; i < num_tokens; ++i) {
      for (int j = 0; j < ELEMENTS_TESTED_PER_TOKEN; ++j) {
        uint16_t expected = float_to_bf16(static_cast<float>((j + 1) * 2));
        uint16_t actual = (combined_output_host[i * hidden + j]);
        if (actual != expected) {
          printf("Combine check failed! Rank %d, token %d, element %d: expected %d, got %d\n",
                 myRank, i, j, expected, actual);
          combine_errors++;
          if (combine_errors >= 5) break; // Limit error output
        }
      }
      if (combine_errors >= 5) break;
    }
    delete[] combined_output_host;
  }

  if (random_mode) {
    printf("Rank %d: Combine flow completed (random mode, checks skipped)\n", myRank);
  } else if (combine_errors == 0) {
    printf("Rank %d: Combine verification PASSED! All %d tokens with %d elements each correctly combined\n",
           myRank, num_tokens, hidden);
  } else {
    printf("Rank %d: Combine verification FAILED with %d errors\n", myRank, combine_errors);
    printf("Rank %d: Exiting test due to combine failure\n", myRank);
    exit(EXIT_FAILURE);
  }

  // Cached mode test: repeat dispatch and combine calls with the same inputs but new outputs
  // Compare results between first and second phase to verify cached mode correctness
  if (cached_mode) {
    // Cached mode is only supported in HT modes (not LL)
    if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
      printf("Rank %d: Error - cached mode is only supported in HT modes (not LL)\n", myRank);
      exit(EXIT_FAILURE);
    }

    printf("Rank %d: Testing cached mode (%s)\n", myRank, algorithm_name);

    // Save first phase dispatch outputs to host for comparison (only if not in random mode)
    uint16_t *first_dispatch_output0 = nullptr;
    float *first_dispatch_output1 = nullptr;
    int64_t *first_dispatch_output2 = nullptr;
    uint16_t *first_combine_output = nullptr;

    if (!random_mode) {
      first_dispatch_output0 = new uint16_t[num_recv_tokens * hidden];
      first_dispatch_output1 = new float[num_recv_tokens * top_k];
      first_dispatch_output2 = new int64_t[num_recv_tokens * top_k];
      CUDACHECK(cudaMemcpy(first_dispatch_output0, outputs[0]->data,
                           num_recv_tokens * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      CUDACHECK(cudaMemcpy(first_dispatch_output1, outputs[1]->data,
                           num_recv_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));
      CUDACHECK(cudaMemcpy(first_dispatch_output2, outputs[2]->data,
                           num_recv_tokens * top_k * sizeof(int64_t), cudaMemcpyDeviceToHost));

      // Save first phase combine output to host for comparison
      first_combine_output = new uint16_t[num_tokens * hidden];
      CUDACHECK(cudaMemcpy(first_combine_output, combined_output.data,
                           num_tokens * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }

    // Allocate new output tensors for second phase dispatch
    ncclNDTensor_t *cached_outputs[1];
    cached_outputs[0] = new ncclNDTensor_t;
    NCCLCHECK(ncclEpTensorCreate(ep_group, cached_outputs[0], 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,
                       static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(hidden)));

    // Allocate new output tensors for second phase combine
    ncclNDTensor_t cached_combined_output;
    NCCLCHECK(ncclEpTensorCreate(ep_group, &cached_combined_output, 2, ncclBfloat16, NCCL_EP_TENSOR_TAG_TOKENS,
                       static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(hidden)));
    ncclNDTensor_t cached_combined_topk_weights;
    NCCLCHECK(ncclEpTensorCreate(ep_group, &cached_combined_topk_weights, 2, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                       static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(top_k)));
    ncclNDTensor_t *cached_combine_outputs[2];
    cached_combine_outputs[0] = &cached_combined_output;
    cached_combine_outputs[1] = &cached_combined_topk_weights;

    // Setup combine inputs with per-received-token topk_weights from dispatch output.
    // HT backward combine expects COMBINE_INPUT_TOPK_WEIGHTS to align with COMBINE_INPUT_TOKENS.

    // Create a new tensor with the correct COMBINE_INPUT tag and copy data from dispatch output.
    ncclNDTensor_t cached_combine_topk_weights_input;
    NCCLCHECK(ncclEpTensorCreate(ep_group, &cached_combine_topk_weights_input, 2, ncclFloat32, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS,
                       static_cast<unsigned int>(num_recv_tokens), static_cast<unsigned int>(top_k)));
    CUDACHECK(cudaMemcpy(cached_combine_topk_weights_input.data, outputs[1]->data,
                         num_recv_tokens * top_k * sizeof(float), cudaMemcpyDeviceToDevice));

    ncclNDTensor_t *cached_combine_inputs[2];
    cached_combine_inputs[0] = &expert_outputs;
    cached_combine_inputs[1] = &cached_combine_topk_weights_input;

    printf("Rank %d: Testing cached mode - second ncclEpDispatch call (send_only=%s)\n",
           myRank, dispatch_send_only ? "true" : "false");
    NCCLCHECK(ncclEpDispatch(ep_handle, inputs, 1, cached_outputs, 1,
      nullptr, 0,
      dispatch_send_only /* send_only */, &dispatch_config, s));

    printf("Rank %d: Testing cached mode - second ncclEpComplete (dispatch)\n", myRank);
    NCCLCHECK(ncclEpComplete(ep_handle, nullptr, s));
    CUDACHECK(cudaStreamSynchronize(s));

    printf("Rank %d: Testing cached mode - second ncclEpCombine call (send_only=%s)\n",
           myRank, combine_send_only ? "true" : "false");
    NCCLCHECK(ncclEpCombine(ep_handle, cached_combine_inputs, 2, cached_combine_outputs, 2,
      nullptr, 0, /* no local_tensors for HT combine */
      combine_send_only /* send_only */, nullptr /* config reserved */, s));

    printf("Rank %d: Testing cached mode - second ncclEpComplete (combine)\n", myRank);
    NCCLCHECK(ncclEpComplete(ep_handle, nullptr, s));
    CUDACHECK(cudaStreamSynchronize(s));

    // Copy second phase outputs to host for comparison (only if not in random mode)
    int cached_dispatch_errors = 0;
    int cached_combine_errors = 0;

    if (!random_mode) {
      uint16_t *second_dispatch_output0 = new uint16_t[num_recv_tokens * hidden];
      CUDACHECK(cudaMemcpy(second_dispatch_output0, cached_outputs[0]->data,
                           num_recv_tokens * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));

      uint16_t *second_combine_output = new uint16_t[num_tokens * hidden];
      CUDACHECK(cudaMemcpy(second_combine_output, cached_combined_output.data,
                           num_tokens * hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      float *second_combine_topk_weights = new float[num_tokens * top_k];
      CUDACHECK(cudaMemcpy(second_combine_topk_weights, cached_combined_topk_weights.data,
                           num_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost));

      // Compare dispatch outputs between first and second phase
      for (unsigned int i = 0; i < num_recv_tokens * hidden; ++i) {
        if (first_dispatch_output0[i] != second_dispatch_output0[i]) {
          if (cached_dispatch_errors < 5) {
            printf("Rank %d: Cached dispatch output0 mismatch at %u: first=%u, second=%u\n",
                   myRank, i, first_dispatch_output0[i], second_dispatch_output0[i]);
          }
          cached_dispatch_errors++;
        }
      }

      // Compare combine outputs between first and second phase
      for (unsigned int i = 0; i < num_tokens * hidden; ++i) {
        if (first_combine_output[i] != second_combine_output[i]) {
          if (cached_combine_errors < 5) {
            printf("Rank %d: Cached combine output mismatch at %u: first=%u, second=%u\n",
                   myRank, i, first_combine_output[i], second_combine_output[i]);
          }
          cached_combine_errors++;
        }
      }

      // Verify combined topk_weights output (should match input topk_weights: 1.0f / top_k)
      float expected_weight = 1.0f / top_k;
      for (int i = 0; i < num_tokens * top_k; ++i) {
        if (second_combine_topk_weights[i] != expected_weight) {
          if (cached_combine_errors < 5) {
            printf("Rank %d: Cached combine topk_weights mismatch at %d: expected=%f, got=%f\n",
                   myRank, i, expected_weight, second_combine_topk_weights[i]);
          }
          cached_combine_errors++;
        }
      }

      delete[] second_dispatch_output0;
      delete[] second_combine_output;
      delete[] second_combine_topk_weights;
    }

    if (random_mode) {
      printf("Rank %d: Cached mode completed (random mode, checks skipped)\n", myRank);
    } else if (cached_dispatch_errors == 0 && cached_combine_errors == 0) {
      printf("Rank %d: Cached mode verification PASSED - dispatch and combine outputs match\n", myRank);
    } else {
      printf("Rank %d: Cached mode verification FAILED - dispatch errors: %d, combine errors: %d\n",
             myRank, cached_dispatch_errors, cached_combine_errors);
      exit(EXIT_FAILURE);
    }

    // Clean up cached mode tensors
    if (!random_mode) {
      delete[] first_dispatch_output0;
      delete[] first_dispatch_output1;
      delete[] first_dispatch_output2;
      delete[] first_combine_output;
    }
    ncclEpTensorDestroy(ep_group, cached_outputs[0]);
    delete cached_outputs[0];
    ncclEpTensorDestroy(ep_group, &cached_combined_output);
    ncclEpTensorDestroy(ep_group, &cached_combined_topk_weights);
    ncclEpTensorDestroy(ep_group, &cached_combine_topk_weights_input);

    printf("Rank %d: Cached mode - second dispatch and combine calls completed successfully\n", myRank);
  }

  // Clean up first phase host buffer (kept for cached mode comparison in HT mode)
  if (first_dispatch_output0_host) delete[] first_dispatch_output0_host;

  // Clean up combine tensors
  ncclEpTensorDestroy(ep_group, &expert_outputs);
  ncclEpTensorDestroy(ep_group, &topk_weights);
  ncclEpTensorDestroy(ep_group, &combined_output);

  NCCLCHECK(ncclEpHandleDestroy(ep_handle));

  NCCLCHECK(ncclEpGroupDestroy(ep_group, s));

  // finalizing NCCL
  ncclCommDestroy(comm);

  delete[] topk_idx_host;
  ncclEpTensorDestroy(ep_group, &topk_idx);
  // Free recv_expert_counter host tensor (uses cudaFreeHost, not cudaFree) only if it was allocated
  if (disable_max_tokens && handle_local_tensors[0] != nullptr) {
    cudaFreeHost(handle_recv_expert_counter.data);
    delete[] handle_recv_expert_counter.strides;
    delete[] handle_recv_expert_counter.sizes;
  }
  ncclEpTensorDestroy(ep_group, inputs[0]);
  delete inputs[0];
  ncclEpTensorDestroy(ep_group, outputs[0]);
  delete outputs[0];
  if (algorithm != NCCL_EP_ALGO_LOW_LATENCY) {
    ncclEpTensorDestroy(ep_group, outputs[1]);
    delete outputs[1];
    ncclEpTensorDestroy(ep_group, outputs[2]);
    delete outputs[2];
  }
  if (algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
    ncclEpTensorDestroy(ep_group, local_tensors[0]);
    delete local_tensors[0];
  }

  // finalizing MPI
  MPICHECK(MPI_Finalize());

  // Needed for cuda-memcheck --leak-check full
  cudaDeviceReset();

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}

