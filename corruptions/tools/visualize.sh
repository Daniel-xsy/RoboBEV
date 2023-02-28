#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/corruptions/project/config/nuscenes_c_sample.py

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/visual_tools/nuscenes_c_sample.py \
${CONFIG}
# -m debugpy --listen 5678 --wait-for-client 