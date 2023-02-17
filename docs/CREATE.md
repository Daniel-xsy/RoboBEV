<img src="../docs/figs/logo2.png" align="right" width="30%">

# Data Preparation

### Configuration

The [`config`](../corruptions/project/config/nuscenes_c.py) file is a [mmcv style](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html) configuration file. Please first [prepare](./DATA_PREPARE.md) the nuScenes datasets and specify the data path in [`config`](../corruptions/project/config/nuscenes_c.py).

To generate corruptted dataset, specify the `corruptions` parameters using the following format:
```python
corruptions = [dict(type='LowLight', easy=2, mid=3, hard=4),
               dict(type='MotionBlur', easy=1, mid=3, hard=5)]
```

Support corruption types include `MotionBlur`, `Fog`, `Snow`, `ColorQuant`, `Brightness`, `LowLight`, `CameraCrash`, `FrameLost`. 

By default, all the corruptions include three different levels (e.g., `easy`,`mid`,`hard`). The number should between 1 and 5.


### Create Script

All the corruption types are defined at [`corruptions.py`](../corruptions/project/mmdet3d_plugin/corruptions.py). You can follow the script to define your own corruption types.

To generate data, run the following command:

```bash
cd corruptions
bash tools/generate_dataset.sh <CONFIG>
```

This might take several days, you can run it parallelly by modifying the [`config`](../corruptions/project/config/nuscenes_c.py) file.


