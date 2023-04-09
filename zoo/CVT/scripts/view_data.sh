python3 scripts/view_data.py \
  data=nuscenes \
  data.dataset_dir=/nvme/konglingdong/models/nuScenes \
  data.labels_dir=/nvme/konglingdong/models/RoboDet/data/cvt/cvt_labels_nuscenes_v2 \
  data.version=v1.0-mini \
  visualization=nuscenes_viz \
  +split=val