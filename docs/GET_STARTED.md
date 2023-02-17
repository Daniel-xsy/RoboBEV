<img src="../docs/figs/logo2.png" align="right" width="30%">

# Get started

### Test Scripts

The test scripts on nuScenes-c are included in [`./corruptions/tools/`](../corruptions/tools). First copy the test tools:

```bash
cp -r ./corruptions/tools ./zoo/<MODEL>
```

To test on nuScenes-c, simply run the following commands:

```bash
cd ./zoo/<MODEL>
bash tools/dist_robust_test.sh <CONFIG> <CHECKPOINT> <GPU_NUM>
```
However, there are a few things to do before you can run the above command successfully.

### Custom Loading

If the original config uses the `LoadMultiViewImageFromFiles` to load images. You can simply add [`custom_loading.py`](../corruptions/project/mmdet3d_plugin/datasets/pipelines/custom_loading.py) to the corresponding folder (e.g., `./zoo/<MODEL>/projects/mmdet3d_plugin/datasets/pipelines/`) and add the `Custom_LoadMultiViewImageFromFiles` in the `__init__.py` in `pipelines/` folder by following line:

```python
from .transform_3d import PadMultiViewImage, NormalizeMultiviewImage
# import custom module
from .custom_loading import Custom_LoadMultiViewImageFromFiles
__all__ = ['PadMultiViewImage', 'NormalizeMultiviewImage', 
# add the custom module into mmdet registry class
'Custom_LoadMultiViewImageFromFiles']
```

If you use your own way to load images, you can modify it to load nuScenes-c data by simply adding three `attribute` (i.e., `corruption`, `severity`, `corruption_root`) to the original `Class` like this:

```python
@PIPELINES.register_module()
class Custom_LoadMultiViewImageFromFiles(object):
    def __init__(self, to_float32=False, color_type='unchanged', corruption=None, severity=None, corruption_root=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        # the following attributes are used for loading nuScenes-c data
        self.corruption = corruption
        self.severity = severity
        self.corruption_root = corruption_root
        if corruption is not None:
            assert severity in ['easy', 'mid', 'hard'], f"Specify a severity of corruption benchmark, now {severity}"
            assert corruption_root is not None, f"When benchmark corruption, specify nuScenes-C root"
```

Then you only need to change the image path from the original to the corruptted one. Here is an example to do so:

```python
def get_corruption_path(corruption_root, corruption, severity, filepath):
    folder, filename = os.path.split(filepath)
    _, subfolder = os.path.split(folder)
    return os.path.join(corruption_root, corruption, severity, subfolder, filename)
```

For more details, you can refer to [`custom_loading.py`](../corruptions/project/mmdet3d_plugin/datasets/pipelines/custom_loading.py) to custom your own `Class`.

### Custom Configuration

After defining the custom module to load images from nuScenes-c, we can modify the configuration file. First, we should modify the origin loading module to the custom one we defined above. We only need to change the test_pipeline from:

```python
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    ...
]
```

to

```python
corruption_root = path/to/nuScenes-c

test_pipeline = [
    dict(type='Custom_LoadMultiViewImageFromFiles', to_float32=True, corruption_root=corruption_root),
    ...
]
```

And then specify the corruption we want to test by adding:

```python
corruptions = ['Snow', 'ColorQuant', 'LowLight']
```

Then, the model can be tested on nuScenes-c successfully.