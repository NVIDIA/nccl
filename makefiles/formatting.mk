#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Prerequisite: $(FILESTOFORMAT) contains the list of files of interest for formatting
# As this file defines a new target (format), it should be included at least after the definition of the
# default target.

ASTYLE_FORMAT_OPTS=-Qv --style=java --indent-after-parens --indent-modifiers --indent-switches --indent-continuation=2 --keep-one-line-blocks --keep-one-line-statements --indent=spaces=2 --lineend=linux --suffix=none
ASTYLEDIR := $(BUILDDIR)/contrib
ASTYLETAR := $(ASTYLEDIR)/astyle.tar.gz
ASTYLEBIN := $(ASTYLEDIR)/astyle/build/gcc/bin/astyle
ASTYLEBLD := $(ASTYLEDIR)/astyle/build/gcc/
ASTYLEVER := 3.1
ASTYLEURL := "https://versaweb.dl.sourceforge.net/project/astyle/astyle/astyle%20$(ASTYLEVER)/astyle_$(ASTYLEVER)_linux.tar.gz"

$(ASTYLEDIR) :
	@mkdir -p $(ASTYLEDIR)

$(ASTYLETAR) : $(ASTYLEDIR)
	@wget -q -O $(ASTYLETAR) $(ASTYLEURL)

$(ASTYLEBLD) : $(ASTYLETAR)
	@cd $(ASTYLEDIR) && tar xzf $(ASTYLETAR)

$(ASTYLEBIN) : $(ASTYLEBLD)
	${MAKE} -C $(ASTYLEBLD)

.PHONY : format
format : $(ASTYLEBIN)
	@$(ASTYLEBIN) $(ASTYLE_FORMAT_OPTS) $(FILESTOFORMAT)
