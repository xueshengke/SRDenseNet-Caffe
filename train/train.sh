#!/usr/bin/env sh
set -e

# run this .sh file in the Caffe root directory

./build/tools/caffe train --solver=examples/SRDenseNet/solver.prototxt --gpu all $@

