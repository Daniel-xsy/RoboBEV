#!/usr/bin/env bash

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m debugpy --listen 5678 --wait-for-client $(dirname "$0")/generate_dataset.py \
$CONFIG \
