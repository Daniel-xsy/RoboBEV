#!/usr/bin/env bash

python ./create_data.py \
nuscenes \
--root-path /nvme/share/data/sets/nuScenes \
--out-dir ../../data/uda \
--version 'city2city' \
--extra-tag nuscenes
# -m debugpy --listen 5680 --wait-for-client 