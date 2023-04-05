torchpack dist-run -np 8 python tools/test.py \
/nvme/konglingdong/models/RoboDet/zoo/BEVFusion/configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml \
/nvme/konglingdong/models/RoboDet/models/BEVFusion/camera-only-det.pth \
--eval bbox