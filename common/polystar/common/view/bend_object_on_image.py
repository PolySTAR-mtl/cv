import cv2
from typing import Tuple

import numpy as np

from polystar.common.models.image import Image
from polystar.common.models.object import Object

_COLORS = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
]  # seaborn.color_palette() * 255

ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def bend_boxed_text_on_image(img: Image, text: str, topleft: Tuple[int, int], color: Tuple[int, int, int]):
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(
        patch,
        text,
        (margin + 1, h - margin - 2),
        FONT,
        TEXT_SCALE,
        WHITE,
        thickness=TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1] : topleft[1] + h, topleft[0] : topleft[0] + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


def bend_object_on_image(image: Image, obj: Object):
    color = _COLORS[obj.type.value]
    cv2.rectangle(image, (obj.x, obj.y), (obj.x + obj.w, obj.y + obj.h), color, 2)

    bend_boxed_text_on_image(image, f"{obj.type.name} ({obj.confidence:.1%})", (obj.x, obj.y), _COLORS[obj.type.value])
    return image
