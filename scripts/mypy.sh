#!/bin/bash

directory=$1

if [ -z "$1" ]
  then
    directory="."
fi

python -m mypy --config-file pyproject.toml $directory
