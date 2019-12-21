import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class Zone:
    def __init__(self, x_min, x_max, y_min, y_max, threshold, active_pixels, image_mask):

        self.pixels = [
            (x, y)
            for x, y in active_pixels
            if (y_min <= y <= y_max) and (x_min <= x <= x_max)
        ]

        self.mean_r = self.get_mean(0, image_mask)
        self.mean_g = self.get_mean(1, image_mask)
        self.mean_b = self.get_mean(2, image_mask)
        self.threshold = threshold

    def get_mean(self, color, img):
        return sum([img[pix[0], pix[1]][color] for pix in self.pixels]) / len(self.pixels)

    def get_means(self, img):
        mr, mg, mb = 0, 0, 0
        for pix in self.pixels:
            p = img[pix[0], pix[1]]
            mr += p[0]
            mg += p[1]
            mb += p[2]

        n_pixels = len(self.pixels)
        return mr/n_pixels, mg/n_pixels, mb/n_pixels

    def is_matching(self, frame: np.ndarray):
        mean_r, mean_g, mean_b = self.get_means(frame)
        return math.sqrt(pow(mean_r - self.mean_r, 2) + pow(mean_g - self.mean_g, 2) + pow(mean_b - self.mean_b, 2)) < self.threshold


class MaskDetector:
    def __init__(self, image_path: Path, zones_params: List[Tuple[int, int, int, int, int]]):
        image_mask = cv2.imread(str(image_path))
        active_px = [
             (a, b)
             for a in range(0, 720) for b in range(0, 1280)
             if (
                image_mask[a, b].any() and
                int(image_mask[a, b][0]) + int(image_mask[a, b][1]) + int(image_mask[a, b][2]) > 50
                )
        ]

        self.zones = [
            Zone(*zone_params, active_px, image_mask)
            for zone_params in zones_params
        ]

    def is_matching(self, frame: np.ndarray):
        return all(
            zone.is_matching(frame)
            for zone in self.zones
        )


robot_view_detector = MaskDetector(
    Path(__file__).parent / 'mask_robot_view.jpg',
    [
        (0, 2000, 20, 70, 20),
        (0, 2000, 270, 370, 20),
        (0, 2000, 510, 770, 20),
    ]
)


def is_image_from_robot_view(frame):
    return robot_view_detector.is_matching(frame)



