#!/usr/bin/env python3
# Substitute version placeholders in nccl.h.in to produce nccl.h.
# Usage: gen_nccl_h.py <input> <output> <major> <minor> <patch> <suffix> <version_code>
import sys

args = sys.argv[1:]
# suffix may be absent when NCCL_SUFFIX is empty (CMake drops empty list elements)
if len(args) == 6:
    src, dst, major, minor, patch, version = args
    suffix = ''
elif len(args) == 7:
    src, dst, major, minor, patch, suffix, version = args
else:
    sys.exit(f"gen_nccl_h.py: expected 6 or 7 args, got {len(args)}: {args}")
d = open(src).read()
d = (d.replace('${nccl:Major}',   major)
      .replace('${nccl:Minor}',   minor)
      .replace('${nccl:Patch}',   patch)
      .replace('${nccl:Suffix}',  suffix)
      .replace('${nccl:Version}', version))
open(dst, 'w').write(d)
