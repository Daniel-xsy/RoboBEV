# NuScenes-c Dataset

This folder includes the script to generate nuScenes-c dataset, the scripts to test robustness on nuScenes-c.

## Generate NuScenes-c

Please follow the official [nuScenes](https://www.nuscenes.org/) to download the full dataset. Run the following command to generate nuScenes-c with corruption type specified at [config](./project/config/nuscenes_c.py).
```shell
cd ./corruptions
bash tools/generate_dataset.sh
```

## Robustness Evaluation

Please copy the [`./tools`](./tools/) folder to `../zoo/${models}` and add [`./project/mmdet3d_plugin`](./project/mmdet3d_plugin/) to the `../zoo/${models}/projects/mmdet3d_plugin/`. **Note**: DO NOT copy the `mmdet3d_plugin` folder directly since it might overwrite the original file.

Then run the following command:
```shell
cd ../zoo/${models}/
bash tools/dist_robust_test.sh ${CONFIG} ${CKPT} ${GPU}
```

