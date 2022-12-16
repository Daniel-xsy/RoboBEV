# BEVDet


 ![Illustrating the performance of the proposed BEVDet on the nuScenes val set](./resources/nds-fps.png)
 
## News
* **2022.11.24** A new branch of bevdet codebase, dubbed dev2.0, is released. dev2.0 includes the following features:
1. support **BEVPoolv2**, whose inference speed is up to **15.1 times** the previous fastest implementation of Lift-Splat-Shoot view transformer. It is also far less memory consumption.
 ![bevpoolv2](./resources/bevpoolv2.png)
 ![bevpoolv2](./resources/bevpoolv2_performance.png)
2. use the origin of ego coordinate system as the center of the receptive field instead of the Lidar's.
3. **support conversion of BEVDet from pytorch to TensorRT.**
4. use the long term temporal fusion as SOLOFusion.
5. train models without CBGS by default.
6. use key frame for temporal fusion.
7. Technique Report [BEVPoolv2](https://arxiv.org/abs/2211.17111) in English and [Blog](https://zhuanlan.zhihu.com/p/586637783) in Chinese.
* [History](./docs/en/news.md)


## Main Results
| Config            | mAP      | NDS     | FPS     |   Model | Log
|--------|----------|---------|--------|-------------|-------|
| [**BEVDet-R50**](configs/bevdet/bevdet-r50.py)       | 27.8     | 32.2    | 18.7    | [google](https://drive.google.com/drive/folders/1Dh2FbEChbfhIaBHimPP4jbibs8XjJ5rx?usp=sharing)   | [google](https://drive.google.com/drive/folders/1Dh2FbEChbfhIaBHimPP4jbibs8XjJ5rx?usp=sharing) 
| [**BEVDet-R50-CBGS**](configs/bevdet/bevdet-r50-cbgs.py)       | 30.7     | 38.2    | 18.7   | [google](https://drive.google.com/drive/folders/1Dh2FbEChbfhIaBHimPP4jbibs8XjJ5rx?usp=sharing)   | [google](https://drive.google.com/drive/folders/1Dh2FbEChbfhIaBHimPP4jbibs8XjJ5rx?usp=sharing) 
| [**BEVDet-R50-4D-Depth-CBGS**](configs/bevdet/bevdet4d-r50-depth-cbgs.py)       | 40.2/40.6#     | 52.3/52.6#    | 16.4  | [google](https://drive.google.com/drive/folders/1Dh2FbEChbfhIaBHimPP4jbibs8XjJ5rx?usp=sharing)   | [google](https://drive.google.com/drive/folders/1Dh2FbEChbfhIaBHimPP4jbibs8XjJ5rx?usp=sharing) 

\# align previous frame bev feature during the view transformation. 
## Inference speed with different backends

| Backend            | 256x704      | 384x1056     | 512x1408    | 640x1760 
|--------|----------|---------|--------|-------------|
|PyTorch        | 37.9    | 64.7   | 105.7   | 154.2  
|TensorRT       | 18.4   | 25.9   | 40.0    | 58.3    
|TensorRT-FP16  | 7.2    | 10.6   | 15.3    | 21.2     
* Evaluate with [**BEVDet-R50**](configs/bevdet/bevdet-r50.py) on a RTX 3090 GPU. We omit the postprocessing, which runs about 14.3 ms with the PyTorch backend.

## Get Started
#### Installation and Data Preparation
1. Please refer to [getting_started.md](docs/en/getting_started.md) for installing BEVDet as mmdetection3d. [Docker](docker/Dockerfile) is recommended for environment preparation.
2. Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for BEVDet by running:
```shell
# python tools/create_data_bevdet.py
```

#### Estimate the inference speed of BEVDet
```shell
# with pre-computation acceleration
python tools/analysis_tools/benchmark.py $config $checkpoint
# 4D with pre-computation acceleration
python tools/analysis_tools/benchmark_sequential.py $config $checkpoint
# view transformer only
python tools/analysis_tools/benchmark_view_transformer.py $config $checkpoint
```

#### Estimate the flops of BEVDet
```shell
python tools/analysis_tools/get_flops.py configs/bevdet/bevdet-r50.py --shape 256 704
```

#### Visualize the predicted result.
* Official implementation. (Visualization locally only)
```shell
python tools/test.py $config $checkpoint --show --show-dir $save-path
```
* Private implementation. (Visualization remotely/locally)
```shell
python tools/test.py $config $checkpoint --format-only --eval-options jsonfile_prefix=$savepath
python tools/analysis_tools/vis.py $savepath/pts_bbox/results_nusc.json
```

#### Convert to TensorRT and test inference speed.
```shell
1. install mmdeploy from https://github.com/HuangJunJie2017/mmdeploy
2. convert to TensorRT
python tools/convert_bevdet_to_TRT.py $config $checkpoint $work_dir --fuse-conv-bn --fp16
3. test inference speed
python tools/analysis_tools/benchmark_trt.py $config $engine
```

## Acknowledgement
This project is not possible without multiple great open-sourced code bases. We list some notable examples below.
* [open-mmlab](https://github.com/open-mmlab) 
* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
* [BEVFusion](https://github.com/mit-han-lab/bevfusion)
* [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

Beside, there are some other attractive works extend the boundary of BEVDet. 
* [BEVerse](https://github.com/zhangyp15/BEVerse)  for multi-task learning.
* [BEVStereo](https://github.com/Megvii-BaseDetection/BEVStereo)  for stero depth estimation.

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{huang2022bevpoolv2,
  title={BEVPoolv2: A Cutting-edge Implementation of BEVDet Toward Deployment},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2211.17111},
  year={2022}
}

@article{huang2022bevdet4d,
  title={BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection},
  author={Huang, Junjie and Huang, Guan},
  journal={arXiv preprint arXiv:2203.17054},
  year={2022}
}

@article{huang2021bevdet,
  title={BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View},
  author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Yun, Ye and Du, Dalong},
  journal={arXiv preprint arXiv:2112.11790},
  year={2021}
}
```
