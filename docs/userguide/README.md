# Building NCCL Documentation

## Quick Start

To package the userguide into a convenient zip file (`build/pkg/doc/nccl-doc_*.zip`):

```bash
nccl$ make pkg.doc.build
```

To make the NCCL html userguide in the build directory (`build/doc/html`):

```bash
nccl$ make -C docs/userguide html
```

The make commands assume the build environment contains Sphinx and a few python
dependencies (see requirements.txt).  In case you need to build a python virtual
environment to manage these dependencies, a small script is provided to do so.  It
creates (`docs/userguide/.venv`) and invokes the build commands after invoking the
virtual environment.

```bash
nccl$ docs/userguide/build_docs.sh
```

By default this script creates the .venv, installs packages acccording to
requirements.txt, and builds the the pkg.doc.build target to package the
docs into a zip file.  The `docs/userguide/.venv` environment is persistent,
but can be removed manually.


## Alternative: CMake

CMake build system also supports building the docs:

```bash
mkdir -p cmake_build && cd cmake_build
cmake .. -DBUILD_DOC_PACKAGE=ON
cmake --build . --target doc_build
```

Output: `cmake_build/doc/html/index.html` and `cmake_build/pkg/doc/nccl-doc_*.zip`

## Additional Targets

Additional Sphinx output formats (other than html) may be built using the makefile (or wrapper script):

```bash
nccl/docs/userguide$ make linkcheck  # Check external links
nccl/docs/userguide$ make latex      # Build LaTeX/PDF
nccl/docs/userguide$ make man        # Build man pages
```

These targets may require additional packages.  For LaTeX targets, we recommend
the following packages on debian-based distros:

    latexmk texlive-latex-extra

The `help` target lists many additional sphinx output formats.
