from .inference import inference_detector_mv3d
from .test import multi_gpu_test_srcn3d, single_gpu_test_srcn3d

__all__ = ['inference_detector_mv3d','single_gpu_test_srcn3d','multi_gpu_test_srcn3d']