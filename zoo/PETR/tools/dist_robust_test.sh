#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/PETR/projects/configs/robust_test/petr_r50dcn_gridmask_p4.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/PETR/petr_r50_p4_1408x512.pth
GPUS=8
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/robust_test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox

# -m debugpy --listen 5678 --wait-for-client 
