import matplotlib.pyplot as plt
import numpy as np

from polystar.common.models.object import Object


def display_object(image: np.ndarray, obj: Object):
    plt.figure(figsize=(16, 12))
    plt.imshow(image)
    color = "red"
    rect = plt.Rectangle((obj.x, obj.y), obj.w, obj.h, linewidth=1, edgecolor=color, fill=False)
    print((obj.x, obj.y), obj.w, obj.h)
    plt.gca().add_patch(rect)
    plt.text(obj.x, obj.y - 2, f"{obj.type.name} ({int(obj.confidence * 100)}%)", color=color)
    plt.show()
