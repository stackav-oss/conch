#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

# Specify CONCH_ENABLE_CUDA_EXT=1 to compare against gpu ref impl (if installed)
# Specify CONCH_BENCH_NO_CSV=1 to print results to stdout instead of file

# Create output directory
benchmark_name="bev_pool_fwd"
benchmark_dir="results/$benchmark_name"
mkdir -p $benchmark_dir

num_points=(
  "1024"
  "8192"
  "16384"
  "100000"
  "600000"
)

for points in ${num_points[@]}; do
  output_file="$benchmark_dir/$points.csv"
  csv_flag="--csv"

  if [ -v CONCH_BENCH_NO_CSV ]; then
    output_file=/dev/stdout
    csv_flag=" "
  fi

  python benchmarks/bev_pool_benchmark.py $csv_flag --num-points $points --compile-ref --cuda-ref > $output_file
done
