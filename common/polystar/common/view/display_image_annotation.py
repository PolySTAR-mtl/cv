from typing import Iterable

import matplotlib.pyplot as plt

from polystar.common.models.image import Image
from polystar.common.models.image_annotation import ImageAnnotation
from polystar.common.models.object import Object
import seaborn as sns


_COLORS = sns.color_palette()


def display_image_with_objects(image: Image, objects: Iterable[Object]):
    plt.figure(figsize=(16, 9))
    plt.imshow(image)
    for obj in objects:
        if obj.confidence >= 0.5:
            color = _COLORS[obj.type.value]
            rect = plt.Rectangle((obj.x, obj.y), obj.w, obj.h, linewidth=2, edgecolor=color, fill=False)
            plt.gca().add_patch(rect)
            plt.text(obj.x, obj.y - 2, f"{obj.type.name} ({int(obj.confidence*100)}%)", color=color, weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def display_image_annotation(image_annotation: ImageAnnotation):
    display_image_with_objects(image_annotation.image, image_annotation.objects)
