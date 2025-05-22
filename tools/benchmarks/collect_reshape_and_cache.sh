#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Need to enable vLLM to compare against vLLM CUDA implementation
export CONCH_ENABLE_VLLM=1
export VLLM_LOGGING_LEVEL=CRITICAL

# Create output directory
benchmark_name="reshape_and_cache"
benchmark_dir="results/$benchmark_name"
mkdir -p $benchmark_dir

num_tokens=(
  "32"
  "64"
  "128"
  "256"
  "512"
)

for tokens in ${num_tokens[@]}; do
  output_file="$benchmark_dir/$tokens.csv"
  python benchmarks/reshape_and_cache_benchmark.py --csv --num-tokens $tokens > $output_file
done
