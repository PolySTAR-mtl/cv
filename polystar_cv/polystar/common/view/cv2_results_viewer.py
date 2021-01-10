import cv2
import numpy as np

from polystar.common.models.image import Image
from polystar.common.view.results_viewer_abc import ResultViewerABC, ColorView

ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]  # seaborn.color_palette() * 255


class CV2ResultViewer(ResultViewerABC):
    def __init__(self, name: str, delay: int = 1, end_keys: str = "q"):
        self.end_keys = [ord(c) for c in end_keys]
        self.delay = delay
        self.name = name
        self._current_image: Image = None
        self.finished = False
        super().__init__(COLORS)

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyWindow(self.name)

    def new(self, image: Image):
        self.height, self.width, _ = image.shape
        self._current_image = image

    def add_text(self, text: str, x: int, y: int, color: ColorView):
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
        w = min(w, self.width - x)  # clip overlay at image boundary
        h = min(h, self.height - y)
        # Overlay the boxed text onto region of interest (roi) in img
        roi = self._current_image[y : y + h, x : x + w, :]
        cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)

    def add_rectangle(self, x: int, y: int, w: int, h: int, color: ColorView):
        cv2.rectangle(self._current_image, (x, y), (x + w, y + h), color, 2)

    def display(self):
        cv2.imshow(self.name, self._current_image)
        self.finished = cv2.waitKey(self.delay) & 0xFF in self.end_keys
