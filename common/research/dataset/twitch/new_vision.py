import math
from pathlib import Path

import cv2

Y_MIN_Z1 = 20
Y_MAX_Z1 = 70
THRESHOLD_ZONE_1 = 45

Y_MIN_Z2 = 270
Y_MAX_Z2 = 370
THRESHOLD_ZONE_2 = 45

Y_MIN_Z3 = 510
Y_MAX_Z3 = 770
THRESHOLD_ZONE_3 = 45

R = 0
G = 1
B = 2


class Zone:
    def __init__(self, y_min, y_max, threshold, active_pixels, image_mask):

        self.y_min = y_min
        self.y_max = y_max
        self.pixels = [
            (pix[0], pix[1])
            for pix in active_pixels
            if self.y_min < pix[1] < self.y_max
        ]

        self.moyenne_r = self.get_moyenne(R, image_mask)
        self.moyenne_g = self.get_moyenne(G, image_mask)
        self.moyenne_b = self.get_moyenne(B, image_mask)
        self.threshold = threshold

    def get_moyenne(self, color, img):
        return sum([img[pix[0], pix[1]][color] for pix in self.pixels]) / len(self.pixels)

    def get_moyennes(self, img):
        mr, mg, mb = 0, 0, 0
        for pix in self.pixels:
            p = img[pix[0], pix[1]]
            mr += p[0]
            mg += p[1]
            mb += p[2]

        leng = len(self.pixels)
        return mr/leng, mg/leng, mb/leng


class Mask:
    def __init__(self, image):
        self.image = cv2.imread(image)     # Passer l'url et l'ouvrir avec cv2
        self.active_px = [                  # Iterer sur tous les pixels de l'image pour trouver ceux qui sont !=  0 0 0
                         (a, b)
                         for a in range(0, 720) for b in range(0, 1280)
                         if (self.image[a, b].any() and
                             (int(self.image[a, b][0]) + int(self.image[a, b][1]) + int(self.image[a, b][2]) > 50))
                        ]

        self.zone1 = Zone(Y_MIN_Z1, Y_MAX_Z1, 20, self.active_px, self.image)
        self.zone2 = Zone(Y_MIN_Z2, Y_MAX_Z2, 20, self.active_px, self.image)
        self.zone3 = Zone(Y_MIN_Z3, Y_MAX_Z3, 20, self.active_px, self.image)


def process_frame_moyennes(frame, mask):

    if(process_zone_moyennes(frame, mask.zone1) and    # Passer à travers les zones de la classe mask, faire la moyenne des pixels
       process_zone_moyennes(frame, mask.zone2) and
       process_zone_moyennes(frame, mask.zone3)):
        return 1
    else:
        return 0


def process_zone_moyennes(frame, zone):                # calculer la diff avec la moyenne des bonnes immages
    moy_r, moy_g, moy_b = zone.get_moyennes(frame)
    if math.sqrt(pow(moy_r - zone.moyenne_r, 2) + pow(moy_g - zone.moyenne_g, 2) + pow(moy_b - zone.moyenne_b, 2)) < zone.threshold:
        return 1
    else:
        return 0


def is_image_from_robot_view(frame):
    return process_frame_moyennes(frame, mask)


mask = Mask(str(Path(__file__).parent / 'mask.jpg'))

