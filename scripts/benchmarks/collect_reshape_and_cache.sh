#!/bin/bash

# Need to enable vLLM to compare against vLLM CUDA implementation
export CONCH_ENABLE_VLLM=1

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
