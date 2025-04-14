#!/bin/bash

wheel_platform=$1

if [ -z "$wheel_platform" ]; then
    output_dir="dist/"
else
    output_dir="dist/$wheel_platform"
    export CONCH_WHEEL_BUILD_PLATFORM=$wheel_platform
fi

python -m build --outdir $output_dir
