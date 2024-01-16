#!/bin/bash

set -e

ASTYLE_FORMAT_OPTS="-Qv --style=java --indent-after-parens --indent-modifiers --indent-switches --indent-continuation=2 --keep-one-line-blocks --keep-one-line-statements --indent=spaces=2 --lineend=linux --suffix=none"
ASTYLEDIR="${PWD}/contrib"

if [ ! -d src/ ]; then
  if [ -f nccl.h.in ]; then
    ASTYLEDIR="${PWD}/../contrib"
  else
    echo "Run from the top level of the NCCL source tree"
    exit 1
  fi
fi

ASTYLEBIN="${ASTYLEDIR}/astyle/build/gcc/bin/astyle"
ASTYLETAR=${ASTYLEDIR}/astyle.tar.gz
ASTYLEBIN=${ASTYLEDIR}/astyle/build/gcc/bin/astyle
ASTYLEBLD=${ASTYLEDIR}/astyle/build/gcc/
ASTYLEVER=3.1
ASTYLEURL="https://versaweb.dl.sourceforge.net/project/astyle/astyle/astyle%20${ASTYLEVER}/astyle_${ASTYLEVER}_linux.tar.gz"

# Install astyle if needed
prev_dir=${PWD}
if [ ! -f $ASTYLEBIN ]; then
  mkdir -p ${ASTYLEDIR}
  wget -q -O ${ASTYLETAR} ${ASTYLEURL}
  cd ${ASTYLEDIR}
  tar xf ${ASTYLETAR}
  cd ${ASTYLEBLD}
  make
  cd ${prev_dir}
fi

FILES=$(echo "${@:1}")
if [ -z "$FILES" ]; then
  FILES=$(find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|gdrwrap.h|nccl.h')
fi

${ASTYLEBIN} ${ASTYLE_FORMAT_OPTS} ${FILES} > /dev/null
