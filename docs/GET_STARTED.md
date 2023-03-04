<img src="../docs/figs/logo2.png" align="right" width="30%">

# Getting Started

All the models in the `./zoo` are ready to run on `nuScenes-C` by running:
```bash
cd ./zoo/<MODEL>
bash tools/dist_robust_test.sh <CONFIG> <CHECKPOINT> <GPU_NUM>
```
The config files are included in `./config/robust_test` folders. If you want to test your own models, please follow the instructions below:

### Test Scripts

The test scripts on `nuScenes-C` are included in [`./corruptions/tools/`](../corruptions/tools). First copy the test tools:

```bash
cp -r ./corruptions/tools ./zoo/<MODEL>
```

To test on `nuScenes-c`, simply run the following commands:

```bash
cd ./zoo/<MODEL>
bash tools/dist_robust_test.sh <CONFIG> <CHECKPOINT> <GPU_NUM>
```
However, there are a few things to do before you can run the above command successfully.

### Custom Loading

If the original config uses  [`LoadMultiViewImageFromFiles`](https://github.com/open-mmlab/mmdetection3d/blob/47285b3f1e9dba358e98fcd12e523cfd0769c876/mmdet3d/datasets/pipelines/loading.py#L11) to load images. You can simply copy [`custom_loading.py`](../corruptions/project/mmdet3d_plugin/datasets/pipelines/custom_loading.py) to the corresponding folder 
```bash
cp ./corruptions/project/mmdet3d_plugin/datasets/pipelines/custom_loading.py ./zoo/<MODEL>/projects/mmdet3d_plugin/datasets/pipelines/
```

Then add the [`Custom_LoadMultiViewImageFromFiles`](https://github.com/Daniel-xsy/RoboDet/blob/25ab276f73bd3253fe3caf605c8ab871d7e52aa9/corruptions/project/mmdet3d_plugin/datasets/pipelines/custom_loading.py#L18) in the `pipelines/__init__.py` to [register](https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html):

```python
from .custom_loading import Custom_LoadMultiViewImageFromFiles
__all__ = ['Custom_LoadMultiViewImageFromFiles']
```

If you use your own way to load images, you can modify it to load `nuScenes-C` data by simply adding three `attribute` (i.e., `corruption`, `severity`, `corruption_root`) to the original class like this:

```python
@PIPELINES.register_module()
class Custom_LoadMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, color_type='unchanged', corruption=None, severity=None, corruption_root=None):
        self.to_float32 = to_float32
        self.color_type = color_type

        ################################################################
        # the following attributes are used for loading nuScenes-c data
        ################################################################
        self.corruption = corruption
        self.severity = severity
        self.corruption_root = corruption_root
        if corruption is not None:
            assert severity in ['easy', 'mid', 'hard'], f"Specify a severity of corruption benchmark, now {severity}"
            assert corruption_root is not None, f"When benchmark corruption, specify nuScenes-C root"
```

Then modify the image path from the `nuScenes` to `nuScenes-C` one, here is a simple example:

```python
def get_corruption_path(corruption_root, corruption, severity, filepath):
    folder, filename = os.path.split(filepath)
    _, subfolder = os.path.split(folder)
    return os.path.join(corruption_root, corruption, severity, subfolder, filename)
```

For more details, please refer to [`custom_loading.py`](../corruptions/project/mmdet3d_plugin/datasets/pipelines/custom_loading.py) to customize for loading `nuScenes-C`.

### Custom Configuration

Then, modify the original loading module to the custom one defined above. Simply replace the loading module from `LoadMultiViewImageFromFiles`:

```python
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    ...
]
```

to `Custom_LoadMultiViewImageFromFiles`:

```python
corruption_root = path/to/nuScenes-c

test_pipeline = [
    dict(type='Custom_LoadMultiViewImageFromFiles', to_float32=True, corruption_root=corruption_root),
    ...
]
```

Lastly, specify the corruption types to be test by adding:

```python
corruptions = ['CameraCrash', 'FrameLost', 'MotionBlur', , 'ColorQuant', 'Brightness', 
                'LowLight', 'Fog', 'Snow']
```
