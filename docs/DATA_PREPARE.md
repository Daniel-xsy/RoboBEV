<img src="../docs/figs/logo2.png" align="right" width="30%">

# Data Preparation

### Overall Structure

```
└── data 
    └── sets
        │── nuscenes
        └── nuscenes-c        
```

### nuScenes

To install the [nuScenes](https://www.nuscenes.org/nuscenes) dataset, download the data, annotations, and other files from https://www.nuscenes.org/download. Unpack the compressed file(s) into `/data/sets/nuscenes` and your folder structure should end up looking like this:

```
└── nuscenes  
    ├── Usual nuscenes folders (i.e. samples, sweep)
    │
    ├── lidarseg
    │   └── v1.0-{mini, test, trainval} <- contains the .bin files; a .bin file 
    │                                      contains the labels of the points in a 
    │                                      point cloud (note that v1.0-test does not 
    │                                      have any .bin files associated with it)
    │
    └── v1.0-{mini, test, trainval}
        ├── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)
        ├── lidarseg.json  <- contains the mapping of each .bin file to the token   
        └── category.json  <- contains the categories of the labels (note that the 
                              category.json from nuScenes v1.0 is overwritten)
```

Please follow the official instructions of each model repo to process the nuScenes dataset. It's recommend to use the **absolute dataset path** when generate the `.pkl` annotation file.

To generate domain-specific annotation, please use the following command to generate the domain annotation files.
```bash
cd ./uda
bash tools/create_data.sh
```
The `domain` config includes `city2city`, `day2night`, and `dry2rain`.

### nuScenes-C

The dataset is now available at [OpenDataLab](https://opendatalab.com/home), you can download the dataset [here](https://opendatalab.com/nuScenes-C). The dataset used in this work only contains the `raw/image/nuScenes-C.tar.gz` file. If you want to use the `raw/pointcloud` part, please refer to [Robo3D](https://github.com/ldkong1205/Robo3D).

Unpack the compressed file(s) into `/data/sets/nuscenes-c` and your folder structure should end up looking like this:

```
└── nuscenes-c  
    ├── Camera
    │   ├── easy <- contains folders the same as
    │   │           in `nuscenes/samples` folder
    │   ├── mid
    │   └── hard
    │
    ├── Frame
    │   ├── easy 
    │   ├── mid
    │   └── hard
    │
    └── ...
```

### References

Please note that you should cite the corresponding paper(s) once you use these datasets.
```bibtex
@inproceedings{caesar2020nuscenes,
    author = {H. Caesar and V. Bankiti and A. H. Lang and S. Vora and V. E. Liong and Q. Xu and A. Krishnan and Y. Pan and G. Baldan and O. Beijbom},
    title = {nuScenes: A Multimodal Dataset for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {11621--11631},
    year = {2020}
}
```
```bibtex
@article{xie2023robobev,
    title = {RoboBEV: Towards Robust Bird's Eye View Perception under Corruptions},
    author = {Xie, Shaoyuan and Kong, Lingdong and Zhang, Wenwei and Ren, Jiawei and Pan, Liang and Chen, Kai and Liu, Ziwei},
    journal = {arXiv preprint arXiv:2304.06719}, 
    year = {2023}
}
```

