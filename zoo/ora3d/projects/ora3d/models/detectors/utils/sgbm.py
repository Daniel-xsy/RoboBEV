import cv2
import numpy as np


def denorm(image, rgb_mean, rgb_std):
    """
        Denormalize a image.
        Args:
            image: np.ndarray normalized [H, W, 3]
            rgb_mean: np.ndarray [3] among [0, 1] image
            rgb_std : np.ndarray [3] among [0, 1] image
        Returns:
            unnormalized image: np.ndarray (H, W, 3) [0-255] dtype=np.uint8
    """
    image = image * rgb_std + rgb_mean
    image[image > 1] = 1
    image[image < 0] = 0
    image *= 255
    return np.array(image, dtype=np.uint8)


def sgbm(rimg1, rimg2):
    rimg1 = cv2.cvtColor(rimg1, cv2.COLOR_RGB2GRAY)
    rimg2 = cv2.cvtColor(rimg2, cv2.COLOR_RGB2GRAY)

    maxd = 48
    window_size = 3
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=maxd,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    lmbda = 50000
    sigma = 1.2
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(rimg1, rimg2)
    dispr = right_matcher.compute(rimg2, rimg1)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, rimg1, None, dispr) / 16.0

    return disparity


class ConvertToFloat(object):
    """
    Converts image data type to float.
    """
    def __call__(self, left_image, right_image=None):
        return left_image.astype(np.float32), right_image if right_image is None else right_image.astype(np.float32)


class CropTop(object):
    def __init__(self, crop_top_index=None, output_height=None):
        if crop_top_index is None and output_height is None:
            print("Either crop_top_index or output_height should not be None, set crop_top_index=0 by default")
            crop_top_index = 0
        if crop_top_index is not None and output_height is not None:
            print("Neither crop_top_index or output_height is None, crop_top_index will take over")
        self.crop_top_index = crop_top_index
        self.output_height = output_height

    def __call__(self, left_image, right_image=None):
        height, width = left_image.shape[0:2]

        if self.crop_top_index is not None:
            h_out = height - self.crop_top_index
            upper = self.crop_top_index
        else:
            h_out = self.output_height
            upper = height - self.output_height
        lower = height

        left_image = left_image[upper:lower]
        if right_image is not None:
            right_image = right_image[upper:lower]

        return left_image, right_image


class Normalize(object):
    """
    Normalize the image
    """
    def __init__(self, mean, stds):
        self.mean = np.array(mean, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)

    def __call__(self, left_image, right_image=None):
        left_image = left_image.astype(np.float32)
        left_image /= 255.0
        left_image -= np.tile(self.mean, int(left_image.shape[2]/self.mean.shape[0]))
        left_image /= np.tile(self.stds, int(left_image.shape[2]/self.stds.shape[0]))
        left_image.astype(np.float32)
        if right_image is not None:
            right_image = right_image.astype(np.float32)
            right_image /= 255.0
            right_image -= np.tile(self.mean, int(right_image.shape[2]/self.mean.shape[0]))
            right_image /= np.tile(self.stds, int(right_image.shape[2]/self.stds.shape[0]))
            right_image = right_image.astype(np.float32)
        return left_image, right_image


def generate_disparity_map(img_left, img_right):
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_stds = np.array([0.229, 0.224, 0.225])

    convert_to_float = ConvertToFloat()
    crop_top = CropTop(0)
    normalize = Normalize(mean=rgb_mean, stds=rgb_stds)

    img_left, img_right = convert_to_float(img_left, img_right)
    img_left, img_right = crop_top(img_left, img_right)
    img_left, img_right = normalize(img_left, img_right)

    img_left = denorm(img_left, rgb_mean, rgb_stds)
    img_right = denorm(img_right, rgb_mean, rgb_stds)

    disparity_map = sgbm(img_left, img_right)

    return disparity_map
