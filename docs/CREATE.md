<img src="../docs/figs/logo2.png" align="right" width="30%">

# Data Preparation

### Corruption Configuration

The [`config`](../corruptions/project/config/nuscenes_c.py) file is a [`mmcv` style](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) configuration file. Please first [prepare](./DATA_PREPARE.md) the nuScenes datasets and specify the data path carefully in [`config`](../corruptions/project/config/nuscenes_c.py). Then specify the `corruptions` parameters using the below format, our nuScenes-C dataset use the following severity: 

```python
corruptions = [dict(type='CameraCrash', easy=2, mid=4, hard=5),
               dict(type='FrameLost', easy=2, mid=4, hard=5),
               dict(type='MotionBlur', easy=2, mid=4, hard=5),
               dict(type='ColorQuant', easy=1, mid=2, hard=3),
               dict(type='Brightness', easy=2, mid=4, hard=5),
               dict(type='LowLight', easy=2, mid=3, hard=4),
               dict(type='Fog', easy=2, mid=4, hard=5),
               dict(type='Snow', easy=1, mid=2, hard=3)
]
```

Support corruption types include `MotionBlur`, `Fog`, `Snow`, `ColorQuant`, `Brightness`, `LowLight`, `CameraCrash`, `FrameLost`.  By default, all the corruptions include three different levels (e.g., `easy`, `mid`, and `hard`). The number should between 1 and 5.


### Corruption Creation

The corruption generation relies on third-party packages, install by running:

```bash
pip install imagecorruptions
```

All the corruption types are defined at [`corruptions.py`](../corruptions/project/mmdet3d_plugin/corruptions.py). You can follow the script to define additional corruption types.

### Generate Dataset

To generate data, run the following command:

```bash
cd corruptions
bash tools/generate_dataset.sh <CONFIG>
```

This might take several days, you can generate different corruption types parallelly to accelerate.


