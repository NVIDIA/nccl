#!/bin/bash

if [ ! -f maint/pre-commit-hook ]; then
    echo "Run this script from the NCCL base directory"
    exit 1
fi

ln -s $PWD/maint/pre-commit-hook .git/hooks/pre-commit
