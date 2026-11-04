# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tide.fsdp's FSDP2 compatibility check (no TideComm needed; one GPU):
  1. on the tested torch, compat_problems(model) is empty;
  2. a stock entry point whose signature changed makes install() raise, naming it, before patching;
  3. with allow_fallback, install() returns None and leaves FSDP untouched;
  4. a missing per-parameter attribute is reported;
  5. another patch's (*args, **kwargs) wrapper is not reported as a mismatch.

torchrun --standalone --nproc_per_node 1 test/test_fsdp_compat.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

import tide.fsdp as tf


def main() -> int:
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Linear(256, 256)).cuda()
    for m in model:
        fully_shard(m)
    fully_shard(model)
    ok = True

    problems = tf.compat_problems(model)
    print("1. compat_problems on torch", torch.__version__, "->", problems or "[]")
    ok &= not problems

    stock = tf._pg.foreach_reduce
    entry = (tf._pg.foreach_all_gather, tf._pg.foreach_all_gather_copy_out, tf._fp.FSDPParam.free_unsharded_param)

    def changed(fsdp_params, unsharded_grads, reduce_scatter_group):  # a future torch with another signature
        raise AssertionError("never called")

    tf._pg.foreach_reduce = changed
    try:
        try:
            tf.install(model, comm=None)
            print("2. install did not raise")
            ok = False
        except RuntimeError as e:
            print("2. install raised:", str(e).splitlines()[1].strip())
            ok &= "foreach_reduce" in str(e)
        ret = tf.install(model, comm=None, allow_fallback=True)
        untouched = (tf._pg.foreach_all_gather, tf._pg.foreach_all_gather_copy_out,
                     tf._fp.FSDPParam.free_unsharded_param) == entry and tf._pg.foreach_reduce is changed
        print("3. allow_fallback ->", ret, "; FSDP untouched:", untouched)
        ok &= ret is None and untouched and tf._adapter is None
        def other_patch(*args, **kwargs):  # another library's generic wrapper: not an incompatibility
            return stock(*args, **kwargs)

        tf._pg.foreach_reduce = other_patch
        problems = tf.compat_problems(model)
        print("5. under a (*args, **kwargs) wrapper ->", problems or "[]")
        ok &= not problems
    finally:
        tf._pg.foreach_reduce = stock

    fp = tf._get_module_fsdp_state(model[0])._fsdp_param_groups[0].fsdp_params[0]
    saved = fp.__dict__.pop("padded_sharded_param_size", None)
    if saved is None:  # a property on this torch: shadow it on the class instead
        cls_attr = type(fp).padded_sharded_param_size
        del type(fp).padded_sharded_param_size
    problems = tf.compat_problems(model)
    if saved is None:
        type(fp).padded_sharded_param_size = cls_attr
    else:
        fp.__dict__["padded_sharded_param_size"] = saved
    print("4. with padded_sharded_param_size removed ->", problems)
    ok &= any("padded_sharded_param_size" in p for p in problems)

    print("PASS" if ok else "FAIL")
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
