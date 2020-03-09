import cv2
import numpy as np

from polystar.common.models.image import Image
from polystar.common.models.object import Object

_COLORS = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
]  # seaborn.color_palette()

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
