<img src="../docs/figs/logo2.png" align="right" width="37%">

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

Please follow the official instructions of each model repo to process the nuScenes dataset. 

### nuScenes-C

This dataset is pending release for a careful check of potential IP issues. If you would like to test the robustness of your model in the current stage, please seek a solution from the following two options:
- Get in touch with us; include the inference code and the model checkpoint and we will evaluate it for you.
- Generate the corruption sets by yourself and evaluate the model performance accordingly. Kindly refer to more details in [this](https://github.com/Daniel-xsy/RoboBEV/blob/master/docs/CREATE.md) page.


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

