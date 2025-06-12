#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Specify CONCH_BENCH_NO_CSV=1 to print results to stdout instead of file

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
  output_file="$benchmark_dir/$seq_len.csv"
  csv_flag="--csv"

  if [ -v CONCH_BENCH_NO_CSV ]; then
    output_file=/dev/stdout
    csv_flag=" "
  fi

  python benchmarks/copy_blocks_benchmark.py $csv_flag --num-mappings $mappings > $output_file
done
