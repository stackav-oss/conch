#!/bin/bash
# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

export VLLM_LOGGING_LEVEL=CRITICAL

for benchmark in ./benchmarks/*_benchmark.py; do
  echo "Running $benchmark ..."
  python $benchmark
  echo ""
done
