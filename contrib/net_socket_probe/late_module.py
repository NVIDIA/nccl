import os,time,ctypes,datetime
from pathlib import Path
import torch
import torch.distributed as dist
rank=int(os.environ['RANK'])
def log(s):print(f'{time.monotonic():.6f} rank={rank} {s}',flush=True)
torch.cuda.set_device(0)
dist.init_process_group('nccl',init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ.get('MASTER_PORT', '35220')}",rank=rank,world_size=2,timeout=datetime.timedelta(seconds=20))
cpu=dist.new_group(backend='gloo')
x=torch.full((16384,),42 if rank==0 else 0,dtype=torch.int32,device='cuda');out=torch.zeros(1,dtype=torch.int32,device='cuda')
lib=ctypes.CDLL('libcuda.so.1');mod=ctypes.c_void_p();func=ctypes.c_void_p()
blob=ctypes.create_string_buffer(open(os.environ.get('CUBIN_PATH', str(Path(__file__).with_suffix('.cubin'))),'rb').read())
def ck(e):
 if e:raise RuntimeError(f'CUDA driver error {e}')
def load():
 log('MODULE_LOAD_BEGIN');ck(lib.cuModuleLoadData(ctypes.byref(mod),blob));ck(lib.cuModuleGetFunction(ctypes.byref(func),mod,b'late_kernel'));log('MODULE_LOAD_END')
preload=os.environ.get('PRELOAD_MODULE','0')=='1'
if preload:load()
log('WARMUP_BEGIN')
work=dist.isend(x,1) if rank==0 else dist.irecv(x,0)
work.wait();torch.cuda.synchronize();dist.barrier(group=cpu);log('WARMUP_DONE')
for iteration in range(int(os.environ.get('REPEATS', '1'))):
 # Change data after warmup so a stale receive cannot pass validation.
 expected=1000+iteration
 x.fill_(expected if rank==0 else 0)
 torch.cuda.synchronize();dist.barrier(group=cpu)
 log(f'ITERATION {iteration}')
 if rank==0:
  time.sleep(1);log('SEND_BEGIN');work=dist.isend(x,1);work.wait();torch.cuda.synchronize();log('SEND_DONE')
 else:
  log('RECV_BEGIN');work=dist.irecv(x,0);work.wait();log('WORK_WAIT_RETURNED')
  if not preload:load()
  torch.cuda.synchronize();log('RECV_DONE');assert torch.all(x==expected).item()
  ptr=ctypes.c_void_p(out.data_ptr());params=(ctypes.c_void_p*1)(ctypes.cast(ctypes.byref(ptr),ctypes.c_void_p));ck(lib.cuLaunchKernel(func,1,1,1,1,1,1,0,None,params,None));torch.cuda.synchronize();assert out.item()==7
  if not preload:ck(lib.cuModuleUnload(mod))
  log('VALIDATED')
 dist.barrier(group=cpu)
dist.destroy_process_group();log('COMPLETE')
