from typing import Tuple

import matplotlib.pyplot as plt

from polystar.models.image import Image
from polystar.view.results_viewer_abc import ColorView, ResultViewerABC

COLORS = [
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
]


class PltResultViewer(ResultViewerABC):
    def __init__(self, name: str, fig_size: Tuple[int, int] = (16, 9)):
        self.name = name
        self.fig_size = fig_size
        self._current_fig = None
        super().__init__(COLORS)

    def new(self, image: Image):
        self._current_fig = plt.figure(figsize=self.fig_size)
        plt.imshow(image)
        plt.title(self.name)
        plt.axis("off")

    def add_text(self, text: str, x: int, y: int, color: ColorView):
        plt.text(x, y - 2, text, color=color, weight="bold")

    def add_rectangle(self, x: int, y: int, w: int, h: int, color: ColorView):
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, fill=False)
        plt.gca().add_patch(rect)

    def display(self):
        plt.tight_layout()
        plt.show()
        plt.close(self._current_fig)
