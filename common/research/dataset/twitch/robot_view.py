import numpy as np
from scipy.spatial import distance
from skimage import io


ref_image = io.imread(f'{__file__}/../mask.jpg')

_MASK = ref_image[:, :, 1] > 50
_REF_IMG_MASKED = ref_image*_MASK[:, :, np.newaxis]
_THRESHOLD = 23


def is_image_from_robot_view(image: np.ndarray) -> bool:
    img_masked = image * _MASK[:, :, np.newaxis]
    return distance.euclidean(img_masked.flatten() / 255, _REF_IMG_MASKED.flatten() / 255) < _THRESHOLD

