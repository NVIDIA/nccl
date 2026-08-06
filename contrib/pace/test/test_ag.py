from ast import parse
import os
import argparse
import torch
import torch.distributed as dist
from pace.ag import CommConfig, AGComm
from utils import init_dist, bench, bench_kineto
import torch.cuda.nvtx as nvtx
from functools import partial
import random
import math
import torch.nn.functional as F
from typing import Union, List
import time
import traceback

def test_main(args: argparse.Namespace, num_local_ranks: int, num_ranks: int, rank: int,
              seed: int, buffer: AGComm, group: dist.ProcessGroup):

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    def check_func(r : torch.Tensor, ans : Union[torch.Tensor, List[torch.Tensor]], in_ : torch.Tensor):
        split_r = buffer.get_split_tensors(in_, r)
        split_ans = []
        if isinstance(ans, torch.Tensor):
            ans = [ans]
        for a in ans:
            va = a.view(num_ranks, -1)
            split_ans.append([va[i] for i in range(num_ranks)])
        is_close = [all([torch.allclose(kr, ka) for kr, ka in zip(sr, sa)]) for sr, sa in zip(split_r, split_ans)]
        if not all(is_close):
            print(f'[rank {rank}]: result mismatch')
            print(traceback.print_stack())
            output_file = f'ag_{rank}.h5'
            torch.save((r, ans, split_r, split_ans), output_file)
            time.sleep(3)
            raise RuntimeError(f'check failed, save to {output_file}')
        # Exercise get_gathered_tensors (sibling of get_split_tensors): outs[i]
        # is the full per-tensor allgather slice of r and must match ans[i].
        gathered_r = buffer.get_gathered_tensors(in_, r)
        for gr, a in zip(gathered_r, ans):
            if not torch.allclose(gr, a):
                print(f'[rank {rank}]: get_gathered_tensors mismatch')
                torch.save((r, ans, gathered_r), f'ag_gathered_{rank}.h5')
                raise RuntimeError('get_gathered_tensors check failed')
   
    
    def test_ag(input_tensor, rand_input):
        fix_answer = torch.ones((input_tensor.numel() * num_ranks), dtype=input_tensor.dtype, device='cuda')
        rand_answer = torch.ones_like(fix_answer)
        single_bytes = input_tensor.numel() * input_tensor.element_size()
        total_bytes = single_bytes * (num_ranks - 1) # 总线数据量
        if not args.no_check:
            dist.all_gather_into_tensor(output_tensor=rand_answer, input_tensor=rand_input, group=group)
            dist.all_gather_into_tensor(output_tensor=fix_answer, input_tensor=input_tensor, group=group)
            
            output_t = torch.zeros_like(fix_answer)

            # normal
            r, _ = buffer.all_gather(input_tensor)
            check_func(r, fix_answer, input_tensor)

            r, _ = buffer.all_gather(input_tensor, output_t)
            torch.cuda.current_stream().wait_event(buffer.get_stream().record_event())
            check_func(r, fix_answer, input_tensor)

            if buffer.config.num_sms > 0:
                out_bf16 = output_t.to(torch.bfloat16)
                imapping = (torch.float32, torch.bfloat16)
                r, _ = buffer.all_gather(input_tensor, out_bf16, input_dtype_mapping=imapping)
                check_func(r, fix_answer.to(torch.bfloat16), input_tensor)

            # async finish
            r, event = buffer.all_gather(input_tensor, async_finish=True)
            event.current_stream_wait()
            check_func(r, fix_answer, input_tensor)

            # random input
            # normal
            r, _ = buffer.all_gather(rand_input)
            check_func(r, rand_answer, rand_input)
            # async finish
            r, event = buffer.all_gather(rand_input, async_finish=True)
            event.current_stream_wait()
            check_func(r, rand_answer, rand_input)

            # previous_event chaining: capture an event on the current compute
            # stream and feed it back as previous_event — the overlap primitive
            # (previously unexercised for AG; mirrors the test_ep pattern).
            pe = buffer.capture()
            r, _ = buffer.all_gather(rand_input, previous_event=pe)
            check_func(r, rand_answer, rand_input)

            # dual tensors
            batch_input = [input_tensor + i * num_ranks for i in range(6)]
            batch_answer = [fix_answer + i * num_ranks for i in range(6)]

            rs, _ = buffer.all_gather(batch_input)
            check_func(rs, batch_answer, in_ = batch_input)
            
            rs, event = buffer.all_gather(batch_input, async_finish=True)
            event.current_stream_wait()
            check_func(rs, batch_answer, in_ = batch_input)

            # list input with a per-tensor out= list (in-place into caller buffers;
            # previously only single-tensor out= was exercised). list-out buffers
            # are separate per-tensor full gathers (not a packed flat result), so
            # check each rs[i] == list_out[i] directly against its answer.
            list_out = [torch.empty(t.numel() * num_ranks, dtype=t.dtype, device='cuda') for t in batch_input]
            rs, _ = buffer.all_gather(batch_input, out=list_out)
            assert rs is list_out, f'[rank {rank}]: list out= did not return the caller buffers'
            for i in range(len(batch_input)):
                assert torch.allclose(rs[i], batch_answer[i]), f'[rank {rank}]: list out= tensor {i} mismatch'
            
            dist.barrier()
            if rank == 0:
                print(f'[rank {rank}]: check passed')

        if not args.no_bench:
            # Performance Testing
            dist.barrier()
            with nvtx.range(f'pace_allgather'):
                t = bench(lambda: buffer.all_gather(input_tensor), post_fn=dist.barrier)[0]
                if rank == 0 or True:
                    print(f'[rank {rank}]: allgather bandwidth: {total_bytes / t / 1e9:.2f} GBps')
                dist.barrier()
            
            def test_func():
                buffer.all_gather(input_tensor)

            def kineto_bench():
                if args.num_sms > 0:
                    k2p = {
                        'allgather_ring_kernel' : 1,
                    }
                else:
                    if num_local_ranks != num_ranks:
                        k2p = {
                            'allgather_zero_sm_cord_kernel' : 1,
                        }
                    else:
                        use_slots = buffer.get_comm_slots(input_tensor)
                        k2p = {
                            'Memcpy DtoD (Device -> Device)' : use_slots * 2,
                            'Memcpy PtoP (Device -> Device)' : (num_local_ranks - 1) * use_slots,
                        }
                def tfunc(bundle):
                    if num_local_ranks != num_ranks:
                        send_t = sum(bundle)
                    else:
                        send_t = (sum(bundle[0]) + sum(bundle[1])) / 2 # it is not accurate actually, they are on two different streams
                    recv_t = 0
                    return send_t, recv_t, send_t + recv_t
                with nvtx.range('pace_allgather'):
                    ret = bench_kineto(test_func, kernel_names=tuple(k2p.keys()), barrier_comm_profiling=True, suppress_kineto_output=True, num_kernels_per_period=k2p, trace_path=f'kineto_{rank}.json' if args.save_kineto and seed == 0 else None)
                if all([v is None for v in ret]):
                    print(f'[rank {rank}]: kineto failed, it is normal if you launch program with nsys')
                else:
                    send_t, recv_t, total_t = tfunc(ret)
                    if rank == 0 or True:
                        print(f'[rank {rank}]: allgather : total time {total_t * 1e6:.2f} us, bandwidth: {total_bytes / total_t / 1e9:.2f} GBps')

            if num_ranks != num_local_ranks:
                kineto_bench()

            if not args.no_compare_baseline:
                with nvtx.range(f'nccl_allgather'):
                    t = bench(lambda: dist.all_gather_into_tensor(output_tensor=rand_answer, input_tensor=rand_input, group=group), post_fn=dist.barrier)[0]
                    print(f'[rank {rank}]: nccl_allgather bandwidth: {total_bytes / t / 1e9:.2f} GBps')

        dist.barrier()
    
    for seq_len in args.seq_len:
        if rank == 0:
            print(f'\n===== Testing seq_len={seq_len} =====')
        input_tensor = torch.ones((seq_len, ), dtype=torch.float32, device='cuda') * rank
        rand_input = torch.randn((seq_len, ), dtype=torch.float32, device='cuda') + 4
        single_bytes = input_tensor.numel() * input_tensor.element_size()
        alg_bytes = single_bytes * num_ranks # 算法数据量
        bus_bytes = single_bytes * (num_ranks - 1) # 总线数据量
        test_ag(input_tensor, rand_input)

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

    config = CommConfig(slot_unroll=args.unroll, nvl_ring_size=args.nvl_ring, rdma_ring_size=args.rdma_ring, num_sms=args.num_sms)
    config.ag_zero_sm.use_ring = args.use_ring
    assert num_local_ranks % args.split_intra == 0
    num_local_ranks = num_local_ranks // args.split_intra
    buffer = AGComm(group, num_local_ranks=num_local_ranks, config=config)
    if rank == 0:
        sm = buffer.runtime.get_shared_memory_bytes()
        print(f'[rank {rank}]: AG shared memory = {sm/1e6:.2f} MB (buffer + {4*1024*1024/1e6:.1f} MB signal)', flush=True)
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

    parser = argparse.ArgumentParser(description='Test PACE allgather')
    parser.add_argument('--num-processes', type=int, default=None,
                       help='Number of processes to spawn (default: torch.cuda.device_count())')
    parser.add_argument('--dim0', type=int, default=None, help='dim 0, default is num_ranks')
    parser.add_argument('--seq-len', type=int, nargs='+', default=[1048576],
                       help='Total sequence length(s) to test (default: [1048576])')
    parser.add_argument("--sep-out-pfx", type=str, default=None, help="split output file prefix for each rank, stored as <prefix>_<rank>.txt, default: None, means mixed stdout")
    parser.add_argument("--no-check", action='store_true', help='skip correction test')
    parser.add_argument("--no-bench", action='store_true', help='skip benchmark')
    parser.add_argument("--repeat", type=int, default=1, help='number of benchmark repeats')
    parser.add_argument("--num-sms", type=int, default=0, help='num_sms: 0 = zero-SM cord path; >0 = SM-resident ring kernel (default: 0)')
    parser.add_argument("--nvl-ring", type=int, default=4, help='nvl ring size')
    parser.add_argument("--rdma-ring", type=int, default=4, help='rdma ring size')
    parser.add_argument("--unroll", type=int, default=32, help='segment length, unrolls of 1024 float4')
    parser.add_argument("--split-intra", type=int, default=1, help='split 1 node into required micro nodes, for simulation use')
    parser.add_argument("--no-compare-baseline", action='store_true', help='do not compare with baseline')
    parser.add_argument("--save-kineto", action='store_true', help='store kineto trace for profiling')
    parser.add_argument("--lid", type=int, default=0, help='local id for direct launch')
    parser.add_argument("--nlid", type=int, default=0, help='number of local processes for direct launch')
    parser.add_argument("--use-ring", action='store_true', help='cord inter-node ring-forward flag (only meaningful when --num-sms 0; num_sms>0 always uses the ring kernel)')
    args = parser.parse_args()

    if args.nlid > 0:
        test_loop(args.lid, args.nlid, args)
    else:
        num_processes = args.num_processes
        if num_processes is None:
            import torch
            num_processes = torch.cuda.device_count()
        torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
