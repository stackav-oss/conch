#!/bin/bash

for benchmark in ./benchmarks/*_benchmark.py; do
  echo "Running $benchmark ..."
  python $benchmark
  echo ""
done
