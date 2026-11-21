# Building NCCL Documentation

`docs/website` is a single Sphinx project containing the NCCL public
documentation.  The Sphinx source root is `docs/website/source/`; the build
files (`Makefile`, `conf.py.in`, `build_docs.sh`, `requirements.txt`) stay in
`docs/website/`.  The source tree contains the user guide (reStructuredText,
under `source/user-guide/`) and the release notes, installation guide, SLA, BSD
license, archives, and notices (MyST Markdown).

The release notes, installation guide, SLA, BSD license, archives, and notices
are maintained in this repository as MyST Markdown.  They were converted from
DITA, and their DITA sources are retired: edit the Markdown here, including
adding each new release to `source/release-notes/` and each newly archived
version to `source/archives/`.

## Quick Start

To package the documentation into a convenient zip file (`build/pkg/doc/nccl-doc_*.zip`):

```bash
nccl$ make pkg.doc.build
```

To make the NCCL html documentation in the build directory (`build/doc/html`):

```bash
nccl$ make -C docs/website html
```

The make commands assume the build environment contains Sphinx and a few python
dependencies (see requirements.txt).  In case you need to build a python virtual
environment to manage these dependencies, a small script is provided to do so.  It
creates (`docs/website/.venv`) and invokes the build commands after invoking the
virtual environment.

```bash
nccl$ docs/website/build_docs.sh
```

By default this script creates the .venv, installs packages according to
requirements.txt, and builds the pkg.doc.build target to package the
docs into a zip file.  The `docs/website/.venv` environment is persistent,
but can be removed manually.


## Alternative: CMake

CMake build system also supports building the docs:

```bash
mkdir -p cmake_build && cd cmake_build
cmake .. -DBUILD_DOC_PACKAGE=ON
cmake --build . --target doc_build
```

Output: `cmake_build/doc/html/index.html` and `cmake_build/pkg/doc/nccl-doc_*.zip`

## PDFs

The project is configured to build one PDF per document
(`NCCL-User-Guide.pdf`, `NCCL-Release-Notes.pdf`,
`NCCL-Installation-Guide.pdf`, `NCCL-SLA.pdf`, `NCCL-BSD.pdf`, and
`NCCL-Archives.pdf`), each ending with the shared notices.

Note: although PDFs are configured, the project does not currently generate or
publish them.  They are not part of `pkg.doc.build` or the CI documentation
build, and the CI documentation image does not include TeX.  To build them
locally (output in `build/doc/latex`):

```bash
nccl$ make -C docs/website latexpdf
```

The PDF build requires `pdflatex`, `latexmk`, and the TeX Live packages used by
Sphinx.  On debian-based distros we recommend:

    latexmk texlive-latex-extra texlive-fonts-recommended

## llms.txt

The HTML build also generates `llms.txt` using the `sphinx-llm` package.  Set
`SKIP_LLMSTXT=1` to build without it when that package is not installed.

## Additional Targets

Additional Sphinx output formats (other than html) may be built using the makefile (or wrapper script):

```bash
nccl/docs/website$ make linkcheck  # Check external links
nccl/docs/website$ make latex      # Build LaTeX sources for the PDFs
nccl/docs/website$ make man        # Build man pages
```

These targets may require additional packages.

The `help` target lists many additional sphinx output formats.
