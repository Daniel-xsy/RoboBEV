torchpack dist-run -np 8 python tools/robust_test.py \
/nvme/konglingdong/models/RoboDet/zoo/BEVFusion/configs/nuscenes-c/det/centerhead/lssfpn/camera/256x704/swint/bevfusion_camera.yaml \
/nvme/konglingdong/models/RoboDet/models/BEVFusion/camera-only-det.pth \
--eval bbox