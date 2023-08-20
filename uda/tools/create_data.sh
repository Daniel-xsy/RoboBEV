#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python /cpfs01/user/xieshaoyuan/code/RoboBEV/uda/tools/create_data.py \
nuscenes \
--root-path /cpfs01/shared/llmit/llmit_hdd/xieshaoyuan/nuScenes \
--out-dir '/cpfs01/user/xieshaoyuan/code/RoboBEV/data/uda_new' \
--version 'v1.0' \
--domain 'city2city' \
--extra-tag nuscenes \
--canbus /cpfs01/shared/llmit/llmit_hdd/xieshaoyuan/nuScenes