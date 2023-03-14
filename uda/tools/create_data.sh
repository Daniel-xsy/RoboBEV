#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/create_data.py \
nuscenes \
--root-path /nvme/share/data/sets/nuScenes \
--out-dir '/nvme/konglingdong/models/RoboDet/data/uda' \
--version 'v1.0' \
--domain 'day2night' \
--extra-tag nuscenes
# -m debugpy --listen 5680 --wait-for-client 