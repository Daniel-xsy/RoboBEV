#!/usr/bin/env bash

PYTHONPATH="/nvme/konglingdong/models/RoboDet/corruptions":$PYTHONPATH \
python $(dirname "$0")/generate_dataset.py \
/nvme/konglingdong/models/RoboDet/corruptions/project/config/nuscenes_c.py \
# -m debugpy --listen 5677 --wait-for-client 
