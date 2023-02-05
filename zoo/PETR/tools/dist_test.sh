#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/PETR/projects/configs/robust_test/petrv2_vovnet_gridmask_p4_800x320.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/PETR/petrv2_vov_p4_800x320.pth
GPUS=1
PORT=${PORT:-29502}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m debugpy --listen 5678 --wait-for-client -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox

# -m debugpy --listen 5678 --wait-for-client 