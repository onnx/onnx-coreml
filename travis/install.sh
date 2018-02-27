#!/bin/bash

source "${0%/*}/setup.sh"

time CMAKE_ARGS='-DUSE_ATEN=OFF -DUSE_OPENMP=OFF' "$top_dir/install-develop.sh"
