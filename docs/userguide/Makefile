# Makefile for Sphinx documentation
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
DOCSRC  := $(abspath ./source)
DOCROOT := $(abspath .)
DOCPREPDIR  := $(BUILDDIR)/doc
PKGDIR  := $(BUILDDIR)/pkg/doc

DOCGEN_IN  := $(shell (cd $(DOCROOT); find $(DOCROOT) -name "*.in" -type f -print -o -path '$(DOCROOT)/.venv' -prune))
DOCGEN_TMP := $(DOCGEN_IN:.in=)
STATIC_IN  := $(shell find $(DOCSRC)/_static $(DOCSRC)/_templates -type f)
DOCGEN := $(patsubst $(DOCSRC)/%,$(DOCPREPDIR)/%,$(DOCGEN_TMP) $(STATIC_IN))
PKG_ARCH   := $(shell uname -m)

$(DOCPREPDIR)/% : $(DOCSRC)/%
	@printf "Copying %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	cp $< $@ || true

$(DOCPREPDIR)/% : $(DOCSRC)/%.in ../../makefiles/version.mk
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    $< > $@


# You can set these variables from the command line.
SPHINXOPTS    = -W --keep-going
SPHINXBUILD   = sphinx-build
PAPER         =

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Consider using "docs/userguide/build_docs.sh" script to  \
make a virtual environment with the preferred version of Sphinx and rtd-theme installed. \
See README.md for details.)
endif

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees -c $(DOCPREPDIR) $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(DOCSRC)
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = -c $(DOCPREPDIR) $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) $(DOCSRC)

.PHONY: help
help:
	@echo
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  texinfo    to make Texinfo files"
	@echo "  info       to make Texinfo files and run them through makeinfo"
	@echo "  gettext    to make PO message catalogs"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  xml        to make Docutils-native XML files"
	@echo "  pseudoxml  to make pseudoxml-XML files for display purposes"
	@echo "  linkcheck  to check all external links for integrity"
	@echo

.PHONY: clean
clean:
	rm -rf $(DOCPREPDIR)/*

.PHONY: html
html: $(DOCGEN)
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(DOCPREPDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(DOCPREPDIR)/html."

.PHONY: dirhtml
dirhtml: $(DOCGEN)
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(DOCPREPDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(DOCPREPDIR)/dirhtml."

.PHONY: singlehtml
singlehtml: $(DOCGEN)
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(DOCPREPDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(DOCPREPDIR)/singlehtml."

.PHONY: pickle
pickle: $(DOCGEN)
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(DOCPREPDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

.PHONY: json
json: $(DOCGEN)
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(DOCPREPDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

.PHONY: htmlhelp
htmlhelp: $(DOCGEN)
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(DOCPREPDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(DOCPREPDIR)/htmlhelp."

.PHONY: qthelp
qthelp: $(DOCGEN)
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(DOCPREPDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(DOCPREPDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/NCCL.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(DOCPREPDIR)/qthelp/NCCL.qhc"

.PHONY: devhelp
devhelp: $(DOCGEN)
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(DOCPREPDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/NCCL"
	@echo "# ln -s $(DOCPREPDIR)/devhelp $$HOME/.local/share/devhelp/NCCL"
	@echo "# devhelp"

.PHONY: epub
epub: $(DOCGEN)
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(DOCPREPDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(DOCPREPDIR)/epub."

.PHONY: latex
latex: $(DOCGEN)
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(DOCPREPDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(DOCPREPDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

.PHONY: latexpdf
latexpdf: $(DOCGEN)
	@which pdflatex >/dev/null 2>&1 || { echo "\n========================================\nError: 'pdflatex' not found.\nInstall with: sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended\n========================================\n"; exit 1; }
	@which latexmk >/dev/null 2>&1 || { echo "\n========================================\nError: 'latexmk' not found.\nInstall with: sudo apt-get install latexmk\n========================================\n"; exit 1; }
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(DOCPREPDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(DOCPREPDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(DOCPREPDIR)/latex."

.PHONY: latexpdfja
latexpdfja: $(DOCGEN)
	@which platex >/dev/null 2>&1 || { echo "\n========================================\nError: 'platex' not found.\nInstall with: sudo apt-get install texlive-lang-japanese\n========================================\n"; exit 1; }
	@which dvipdfmx >/dev/null 2>&1 || { echo "\n========================================\nError: 'dvipdfmx' not found.\nInstall with: sudo apt-get install texlive-extra-utils\n========================================\n"; exit 1; }
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(DOCPREPDIR)/latex
	@echo "Running LaTeX files through platex and dvipdfmx..."
	$(MAKE) -C $(DOCPREPDIR)/latex all-pdf-ja
	@echo "pdflatex finished; the PDF files are in $(DOCPREPDIR)/latex."

.PHONY: text
text: $(DOCGEN)
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(DOCPREPDIR)/text
	@echo
	@echo "Build finished. The text files are in $(DOCPREPDIR)/text."

.PHONY: man
man: $(DOCGEN)
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(DOCPREPDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(DOCPREPDIR)/man."

.PHONY: texinfo
texinfo: $(DOCGEN)
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(DOCPREPDIR)/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in $(DOCPREPDIR)/texinfo."
	@echo "Run \`make' in that directory to run these through makeinfo" \
	      "(use \`make info' here to do that automatically)."

.PHONY: info
info: $(DOCGEN)
	@which makeinfo >/dev/null 2>&1 || { echo "\n========================================\nError: 'makeinfo' not found.\nInstall with: sudo apt-get install texinfo\n========================================\n"; exit 1; }
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(DOCPREPDIR)/texinfo
	@echo "Running Texinfo files through makeinfo..."
	$(MAKE) -C $(DOCPREPDIR)/texinfo info
	@echo "makeinfo finished; the Info files are in $(DOCPREPDIR)/texinfo."

.PHONY: gettext
gettext: $(DOCGEN)
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(DOCPREPDIR)/locale
	@echo
	@echo "Build finished. The message catalogs are in $(DOCPREPDIR)/locale."

.PHONY: changes
changes: $(DOCGEN)
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(DOCPREPDIR)/changes
	@echo
	@echo "The overview file is in $(DOCPREPDIR)/changes."

.PHONY: linkcheck
linkcheck: $(DOCGEN)
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(DOCPREPDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(DOCPREPDIR)/linkcheck/output.txt."

.PHONY: doctest
doctest: $(DOCGEN)
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(DOCPREPDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(DOCPREPDIR)/doctest/output.txt."

.PHONY: coverage
coverage: $(DOCGEN)
	$(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) $(DOCPREPDIR)/coverage
	@echo "Testing of coverage in the sources finished, look at the " \
	      "results in $(DOCPREPDIR)/coverage/python.txt."

.PHONY: xml
xml: $(DOCGEN)
	$(SPHINXBUILD) -b xml $(ALLSPHINXOPTS) $(DOCPREPDIR)/xml
	@echo
	@echo "Build finished. The XML files are in $(DOCPREPDIR)/xml."

.PHONY: pseudoxml
pseudoxml: $(DOCGEN)
	$(SPHINXBUILD) -b pseudoxml $(ALLSPHINXOPTS) $(DOCPREPDIR)/pseudoxml
	@echo
	@echo "Build finished. The pseudo-XML files are in $(DOCPREPDIR)/pseudoxml."
