#!/bin/bash

export VLLM_LOGGING_LEVEL=CRITICAL

for benchmark in ./benchmarks/*_benchmark.py; do
  echo "Running $benchmark ..."
  python $benchmark
  echo ""
done
