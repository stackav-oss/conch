#!/bin/bash

export VLLM_CONFIGURE_LOGGING=0

for benchmark in ./benchmarks/*_benchmark.py; do
  echo "Running $benchmark ..."
  python $benchmark
  echo ""
done
