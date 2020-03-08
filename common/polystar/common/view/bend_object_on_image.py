import cv2
import numpy as np
import seaborn as sns

from polystar.common.models.image import Image
from polystar.common.models.object import Object

_COLORS = sns.color_palette()
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def bend_object_on_image(image: Image, obj: Object):
    assert image.dtype == np.uint8
    img_h, img_w, _ = image.shape
    margin = 3
    size = cv2.getTextSize(obj.type.name, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = _COLORS[obj.type.value]
    cv2.putText(
        patch,
        obj.type.name,
        (margin + 1, h - margin - 2),
        FONT,
        TEXT_SCALE,
        WHITE,
        thickness=TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), BLACK, thickness=1)
    w = min(w, img_w - obj.x)  # clip overlay at image boundary
    h = min(h, img_h - obj.y)
    # Overlay the boxed text onto region of interest (roi) in img
    roi = image[obj.y : obj.y + h, obj.x : obj.x + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return image
