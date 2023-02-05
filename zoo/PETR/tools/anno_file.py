## Convert data path of 
## [mmdet3d_nuscenes_30f_infos_val.pkl](https://drive.google.com/drive/folders/1_C2yuh51ROF3UzId4L1itwGQVUeVUxU6?usp=sharing) 
## to specified one.


import pickle
import os
import mmcv

anno_path = '/nvme/konglingdong/models/RoboDet/data/mmdet3d_nuscenes_30f_infos_val.pkl'
data_root = '/nvme/share/data/sets/nuScenes/samples'
save_path = '/nvme/konglingdong/models/RoboDet/data/mmdet3d_nuscenes_30f_infos_val_custom.pkl'

def main():

    with open(anno_path, 'rb') as f:
        anno = pickle.load(f)
    
    prog_bar = mmcv.ProgressBar(len(anno['infos']))
    for i in range(len(anno['infos'])):
        for sensor in anno['infos'][i]['cams'].keys():
            filename = os.path.split(anno['infos'][i]['cams'][sensor]['data_path'])[1]
            anno['infos'][i]['cams'][sensor]['data_path'] = os.path.join(data_root, sensor, filename)
        prog_bar.update()
    
    with open(save_path, 'wb') as f:
        pickle.dump(anno, f)


if __name__=='__main__':
    main()