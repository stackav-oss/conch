#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Specify CONCH_BENCH_NO_CSV=1 to print results to stdout instead of file
# Specify CONCH_ENABLE_VLLM=1 to compare against FlashAttnVarlenFunc

# Disable vLLM logging
export VLLM_LOGGING_LEVEL=CRITICAL

# Create output directory
benchmark_name="varlen_attention_prefill"
benchmark_dir="results/$benchmark_name"
mkdir -p $benchmark_dir

sequence_lengths=(
  "32"
  "64"
  "128"
  "256"
  "512"
  "1024"
  "2048"
  "4096"
  "8192"
)

for seq_len in ${sequence_lengths[@]}; do
  output_file="$benchmark_dir/$seq_len.csv"
  csv_flag="--csv"

  if [ -v CONCH_BENCH_NO_CSV ]; then
    output_file=/dev/stdout
    csv_flag=" "
  fi

  # Llama-3.1-405B attention layer configuration
  python benchmarks/varlen_attention_benchmark.py $csv_flag --batch-size 128 --num-query-heads 128 --num-kv-heads 8 --head-dim 128 --causal --seq-len $seq_len > $output_file
done
