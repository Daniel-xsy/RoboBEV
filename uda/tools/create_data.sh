#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/create_data.py \
nuscenes \
--root-path /data/nuScenes \
--out-dir '/data/nuScenes' \
--version 'v1.0' \
--domain 'dry2rain' \
--extra-tag nuscenes