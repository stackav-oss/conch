#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Specify CONCH_ENABLE_VLLM=1 to compare against vLLM ref impl (if installed)
# Specify CONCH_BENCH_NO_CSV=1 to print results to stdout instead of file

# Need to enable vLLM to compare against vLLM CUDA implementation
export VLLM_LOGGING_LEVEL=CRITICAL

# Create output directory
benchmark_name="reshape_and_cache"
benchmark_dir="results/$benchmark_name"
mkdir -p $benchmark_dir

num_tokens=(
  "64"
  "128"
  "512"
  "2048"
  "8192"
  "32768"
)

for tokens in ${num_tokens[@]}; do
  output_file="$benchmark_dir/$seq_len.csv"
  csv_flag="--csv"

  if [ -v CONCH_BENCH_NO_CSV ]; then
    output_file=/dev/stdout
    csv_flag=" "
  fi

  python benchmarks/reshape_and_cache_benchmark.py $csv_flag --num-tokens $tokens > $output_file
done
