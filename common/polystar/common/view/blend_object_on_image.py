import cv2

from polystar.common.models.image import Image
from polystar.common.models.object import Object
from polystar.common.view.blend_text_on_image import blend_boxed_text_on_image

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


def blend_object_on_image(image: Image, obj: Object):
    color = _COLORS[obj.type.value]
    cv2.rectangle(image, (obj.x, obj.y), (obj.x + obj.w, obj.y + obj.h), color, 2)

    blend_boxed_text_on_image(image, f"{obj.type.name} ({obj.confidence:.1%})", (obj.x, obj.y), _COLORS[obj.type.value])
    return image
