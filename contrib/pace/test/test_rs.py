from ast import parse
import os
import argparse
import torch
import torch.distributed as dist
from pace.rs import RSComm, CommConfig
from utils import init_dist, bench, bench_kineto
import torch.cuda.nvtx as nvtx
from functools import partial
import random
import math
import torch.nn.functional as F
from typing import Union, List
import time

def _get_dim0_padded_size(tensor_size: torch.Size, dim0_factor: int) -> torch.Size:
    padded_dim0 = math.ceil(tensor_size[0] / dim0_factor) * dim0_factor
    return torch.Size([padded_dim0]) + tensor_size[1:]

class BgWorkload:
    """Memory-bandwidth heavy background workload, launched on current (default) stream."""
    def __init__(self, size_mb: int, iters: int):
        n = max(1, size_mb * 1024 * 1024 // 2)  # bf16 elements
        self.a = torch.randn(n, dtype=torch.bfloat16, device='cuda')
        self.b = torch.randn(n, dtype=torch.bfloat16, device='cuda')
        self.iters = iters

    def launch(self):
        # memory-bound element-wise ops, purely stresses HBM bandwidth
        for _ in range(self.iters):
            self.a.add_(self.b)


def fsdp_rs(unsharded_grads : Union[torch.Tensor, List [torch.Tensor]], world_size : int, group : dist.ProcessGroup, reduce_dtype : torch.dtype = torch.float32, reduce_op : dist.ReduceOp.RedOpType = dist.ReduceOp.SUM):
    unsharded_grads = [unsharded_grads] if isinstance(unsharded_grads, torch.Tensor) else unsharded_grads
    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_scatter_input = torch.empty(
        (reduce_scatter_input_numel,), dtype=reduce_dtype, device='cuda'
    )
    reduce_scatter_output = torch.empty((reduce_scatter_output_numel,), dtype=reduce_dtype, device='cuda')
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    torch._chunk_cat(
        unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
    )
    dist.reduce_scatter_tensor(reduce_scatter_output, reduce_scatter_input, op=reduce_op, group=group)
    return reduce_scatter_output

def test_main(args: argparse.Namespace, num_local_ranks: int, num_ranks: int, rank: int,
              seed: int, buffer: RSComm, group: dist.ProcessGroup):

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    premul_factor = 1.0 / 4 if args.redop == 'premul_sum' else 1.0
    if args.redop == 'sum':
        red_op = dist.ReduceOp.SUM
    elif args.redop == 'premul_sum':
        red_op = dist._make_nccl_premul_sum(premul_factor)
    elif args.redop == 'avg':
        red_op = dist.ReduceOp.AVG
    def check_func(r : torch.Tensor, ans : Union[torch.Tensor, List[torch.Tensor]], in_ : torch.Tensor):
        split_r = buffer.get_split_tensors(in_, r)
        split_ans = buffer.get_split_tensors(in_, ans, tight=True) if isinstance(ans, torch.Tensor) else ans
        is_close = [torch.allclose(sr, sa) for sr, sa in zip(split_r, split_ans)]
        if not all(is_close):
            print(f'[rank {rank}]: result mismatch')
            output_file = f'rs_{rank}.h5'
            torch.save((r, ans, split_r, split_ans, in_), output_file)
            time.sleep(8)
            raise RuntimeError(f'check failed, save to {output_file}')
    fsdp_kwargs = {'world_size' : num_ranks, 'reduce_op': red_op, 'group' : group}
    rs_kwargs = {'red_op' : red_op, 'extra_mul' : args.extra_mul, 'extra_post_mul' : args.extra_post_mul}
    def test_rs(input_tensor, rand_input):
        fix_answer = fsdp_rs(input_tensor, **fsdp_kwargs) * args.extra_mul * args.extra_post_mul
        if not args.no_check:
            rand_answer = fsdp_rs(rand_input, **fsdp_kwargs) * args.extra_mul * args.extra_post_mul
            # normal
            r, _ = buffer.reduce_scatter(input_tensor, **rs_kwargs)
            check_func(r, fix_answer, input_tensor)
            _, _ = buffer.reduce_scatter(input_tensor, out=r, **rs_kwargs)
            check_func(r, fix_answer, input_tensor)
            bf16_r, _ = buffer.reduce_scatter(input_tensor, out_dtype=torch.bfloat16, **rs_kwargs)
            check_func(bf16_r, fix_answer.to(torch.bfloat16), input_tensor)
            # async finish
            r, event = buffer.reduce_scatter(input_tensor, async_finish=True, **rs_kwargs)
            event.current_stream_wait()
            check_func(r, fix_answer, input_tensor)

            # random input
            # normal
            r, _ = buffer.reduce_scatter(rand_input, **rs_kwargs)
            check_func(r, rand_answer, rand_input)
            if args.dtype_size == 2:
                # float precision error may cause a4p rs and nccl rs get different bf16 result...
                # if input is bf16, error will be controlled
                bf16_r, _ = buffer.reduce_scatter(rand_input, out_dtype=torch.bfloat16, **rs_kwargs)
                check_func(bf16_r, rand_answer.to(torch.bfloat16), rand_input)
            # async finish
            r, event = buffer.reduce_scatter(rand_input, async_finish=True, **rs_kwargs)
            event.current_stream_wait()
            check_func(r, rand_answer, rand_input)

            # previous_event chaining (overlap primitive; previously unexercised for RS).
            pe = buffer.capture()
            r, _ = buffer.reduce_scatter(rand_input, previous_event=pe, **rs_kwargs)
            check_func(r, rand_answer, rand_input)

            # dual tensors
            batch_input = [input_tensor + i for i in range(6)]
            batch_answer = [fix_answer + num_ranks * i for i in range(6)]
            def check_batch(rs):
                cnt = 0
                for ans in batch_answer:
                    check_func(rs[cnt:cnt+ans.numel()], ans)
                    cnt += ans.numel()

            rs, _ = buffer.reduce_scatter(batch_input, **rs_kwargs)
            check_func(rs, torch.cat(batch_answer), in_ = batch_input[0])
            rs, event = buffer.reduce_scatter(batch_input, async_finish=True, **rs_kwargs)
            event.current_stream_wait()
            check_func(rs, torch.cat(batch_answer), in_ = batch_input[0])

            dist.barrier()
            if rank == 0:
                print(f'[rank {rank}]: check passed')

        if not args.no_bench:
            # Performance Testing
            dist.barrier()
            with nvtx.range(f'gin_reduce_scatter'):
                t = bench(lambda: buffer.reduce_scatter(input_tensor, out=fix_answer, **rs_kwargs), post_fn=dist.barrier)[0]
                if rank == 0 or True:
                    print(f'[rank {rank}]: reduce-scatter bandwidth: {total_bytes / t / 1e9:.2f} GBps')
                dist.barrier()

            call_times = buffer.get_call_times(input_tensor)
            call_times = sum(call_times)
            def test_func():
                buffer.reduce_scatter(input_tensor, out=fix_answer, **rs_kwargs)

            def kineto_bench():
                klist = ['reduce_scatter']
                k2p = {}
                for k in klist:
                    k2p[k] = call_times
                def tfunc(bundle):
                    send_t = sum(x if isinstance(x, (int, float)) else sum(x) for x in bundle)
                    recv_t = 0
                    return send_t, recv_t, send_t + recv_t
                with nvtx.range('gin_reduce_scatter'):
                    ret = bench_kineto(test_func, kernel_names=tuple(k2p.keys()), barrier_comm_profiling=True, suppress_kineto_output=True, num_kernels_per_period=k2p, trace_path=f'kineto_{rank}.json' if args.save_kineto and seed == 0 else None)
                if all([v is None for v in ret]):
                    print('kineto failed, it is normal if you launch program with nsys')
                else:
                    send_t, recv_t, total_t = tfunc(ret)
                    if rank == 0 or True:
                        print(f'[rank {rank}]: reduce_scatter : total time {total_t * 1e6:.2f} us, bandwidth: {total_bytes / total_t / 1e9:.2f} GBps')
            if not args.skip_kineto:
                kineto_bench()

            if not args.no_compare_baseline:
                with nvtx.range(f'nccl_reduce_scatter'):
                    t = bench(lambda: dist.reduce_scatter_tensor(fix_answer, input_tensor, group=group), post_fn=dist.barrier)[0]
                    print(f'[rank {rank}]: nccl_reduce_scatter bandwidth: {total_bytes / t / 1e9:.2f} GBps')
                with nvtx.range(f'nccl_fsdp_rs'):
                    t = bench(lambda: fsdp_rs(input_tensor, **fsdp_kwargs), post_fn=dist.barrier)[0]
                    print(f'[rank {rank}]: nccl_fsdp_rs bandwidth: {total_bytes / t / 1e9:.2f} GBps')

        dist.barrier()

    
    dtype = torch.float32 if args.dtype_size == 4 else torch.bfloat16
    # Iterate over all sequence lengths
    dim0 = num_ranks if args.dim0 is None else args.dim0
    for seq_len in args.seq_len:
        if rank == 0:
            print(f'\n===== Testing seq_len={seq_len} =====')
        input_tensor = torch.ones((dim0, seq_len), dtype=dtype, device='cuda') * rank
        rand_input = torch.randn((dim0, seq_len), dtype=dtype, device='cuda') + 4
        total_bytes = input_tensor.numel() * dtype.itemsize
        test_rs(input_tensor, rand_input)

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

    config = CommConfig(slot_unroll=args.unroll, nvl_ring_size=args.nvl_ring, rdma_ring_size=args.rdma_ring, num_sms=args.num_sms, use_wg=args.use_wg)
    assert num_local_ranks % args.split_intra == 0
    num_local_ranks = num_local_ranks // args.split_intra
    buffer = RSComm(group, num_local_ranks=num_local_ranks, config=config)
    if rank == 0:
        sm = buffer.runtime.get_shared_memory_bytes()
        print(f'[rank {rank}]: RS shared memory = {sm/1e6:.2f} MB (buffer + {4*1024*1024/1e6:.1f} MB signal)', flush=True)
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

    parser = argparse.ArgumentParser(description='Test PACE reduce-scatter')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of processes to spawn (default: torch.cuda.device_count())')
    parser.add_argument('--dim0', type=int, default=None, help='dim 0, default is num_ranks')
    parser.add_argument('--seq-len', type=int, nargs='+', default=[1048576],
                       help='Total sequence length(s) to test (default: [1048576])')
    parser.add_argument("--sep-out-pfx", type=str, default=None, help="split output file prefix for each rank, stored as <prefix>_<rank>.txt, default: None, means mixed stdout")
    parser.add_argument("--no-check", action='store_true', help='skip correction test')
    parser.add_argument("--no-bench", action='store_true', help='skip benchmark')
    parser.add_argument("--repeat", type=int, default=1, help='number of benchmark repeats')
    parser.add_argument("--dtype-size", type=int, default=4, choices=[2, 4], help='data type size in bytes, 4 : float32, 2: bfloat16. (default: 2)')
    parser.add_argument("--num-sms", type=int, default=32, help='number of SMs to use for PACE kernels (default: 32)')
    parser.add_argument("--nvl-ring", type=int, default=4, help='nvl ring size')
    parser.add_argument("--rdma-ring", type=int, default=4, help='rdma ring size')
    parser.add_argument("--unroll", type=int, default=32, help='segment length, unrolls of 1024 float4')
    parser.add_argument("--split-intra", type=int, default=1, help='split 1 node into required micro nodes, for simulation use')
    parser.add_argument("--no-compare-baseline", action='store_true', help='do not compare with baseline')
    parser.add_argument("--save-kineto", action='store_true', help='store kineto trace for profiling')
    parser.add_argument("--redop", type=str, choices=['sum', 'avg', 'premul_sum'], default='sum', help='reduce type')
    parser.add_argument("--extra-mul", type=float, default=1.0, help='extra multiplier for SUM reduce type')
    parser.add_argument("--extra-post-mul", type=float, default=1.0, help='extra post multiplier for SUM reduce type')
    parser.add_argument("--lid", type=int, default=0, help='local id for direct launch')
    parser.add_argument("--nlid", type=int, default=0, help='number of local processes for direct launch')
    parser.add_argument("--skip-kineto", action='store_true', help='skip kineto tracing')
    parser.add_argument("--use-wg", action='store_true', help='use warp group for reduce scatter')
    args = parser.parse_args()

    if args.nlid > 0:
        test_loop(args.lid, args.nlid, args)
    else:
        num_processes = args.num_processes
        if num_processes is None:
            import torch
            num_processes = torch.cuda.device_count()
        torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
