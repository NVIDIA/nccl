# Maintenance Scripts for NCCL

This directory contains developer maintenance scripts that are not part of the
normal NCCL build. Run the scripts from the repository root unless a script says
otherwise.

## Formatting

NCCL uses `clang-format` for C/C++/CUDA source formatting. The formatting
scripts operate on source files with these suffixes:

```text
.cc .cu .cuh .h .h.in
```

Paths listed in the repository-level `.clang-format-ignore` file are skipped.

### `run-clang-format.sh`

Apply `clang-format` to NCCL source files.

```sh
./maint/run-clang-format.sh
./maint/run-clang-format.sh src/transport
./maint/run-clang-format.sh src/transport/net_ib/common.cc
```

By default, the script formats files under `src/`. If a file or directory path
is provided, it formats only that file or files under that directory
recursively.

Useful options:

```sh
./maint/run-clang-format.sh --list [path]
./maint/run-clang-format.sh --diff [path]
```

The script requires `clang-format` version 22 or newer. Set `CLANG_FORMAT` to
use a non-default binary:

```sh
CLANG_FORMAT=/path/to/clang-format ./maint/run-clang-format.sh --diff
```

## Git Integration

LLVM provides `git-clang-format`, a Git integration script for formatting only
changed lines. Download it from:

```text
https://github.com/llvm/llvm-project/blob/main/clang/tools/clang-format/git-clang-format
```

Follow the instructions in that script: put it somewhere on your `PATH`, make it
executable, and then use `git clang-format`.

Common examples:

```sh
git clang-format
git clang-format --staged
git clang-format main
git clang-format --diff main
```

Run `git clang-format -h` for the full option list.
