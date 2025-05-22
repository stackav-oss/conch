#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Need to enable vLLM to compare against vLLM CUDA implementation
export CONCH_ENABLE_VLLM=1
export VLLM_LOGGING_LEVEL=CRITICAL

# Create output directory
benchmark_name="copy_blocks"
benchmark_dir="results/$benchmark_name"
mkdir -p $benchmark_dir

num_mappings=(
  "32"
  "64"
  "128"
  "256"
  "512"
)

for mappings in ${num_mappings[@]}; do
  output_file="$benchmark_dir/$mappings.csv"
  python benchmarks/copy_blocks_benchmark.py --csv --num-mappings $mappings > $output_file
done
