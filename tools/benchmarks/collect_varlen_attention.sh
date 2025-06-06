#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Need to enable vLLM to compare against FlashAttnWithKVCache
export CONCH_ENABLE_VLLM=1
export VLLM_LOGGING_LEVEL=CRITICAL

# Create output directory
benchmark_name="varlen_attention"
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
  "16384"
  "32768"
)

for seq_len in ${sequence_lengths[@]}; do
  output_file="$benchmark_dir/$seq_len.csv"
  # Llama-3.1-405B attention layer configuration
  python benchmarks/varlen_attention_benchmark.py --csv --batch-size 4 --num-query-heads 128 --num-kv-heads 8 --head-dim 128 --causal --seq-len $seq_len > $output_file
done
