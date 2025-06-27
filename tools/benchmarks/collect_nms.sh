#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Specify CONCH_ENABLE_TORCHVISION=1 to compare against torchvision gpu ref impl (if installed)
# Specify CONCH_BENCH_NO_CSV=1 to print results to stdout instead of file

# Create output directory
benchmark_name="nms"
benchmark_dir="results/$benchmark_name"
mkdir -p $benchmark_dir

num_boxes=(
  "512"
  "2048"
  "8192"
)

for boxes in ${num_boxes[@]}; do
  output_file="$benchmark_dir/$seq_len.csv"
  csv_flag="--csv"

  if [ -v CONCH_BENCH_NO_CSV ]; then
    output_file=/dev/stdout
    csv_flag=" "
  fi

  python benchmarks/nms_benchmark.py $csv_flag --num-boxes $boxes > $output_file
done
