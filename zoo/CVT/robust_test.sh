corruptions="CameraCrash FrameLost ColorQuant MotionBlur Brightness LowLight Fog Snow"
severities="easy mid hard"
for corruption in ${corruptions} 
do  
  for severity in ${severities}
  do
  python3 scripts/train.py \
    +experiment=cvt_nuscenes_vehicle \
    data.dataset_dir=/nvme/konglingdong/data/sets/nuScenes-c/${corruption}/${severity}/ \
    data.labels_dir=/nvme/konglingdong/models/RoboDet/data/cvt/cvt_labels_nuscenes_v2
  done
done