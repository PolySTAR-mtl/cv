import cv2
from pathlib import Path
import numpy
import math
import time
from research.constants import TWITCH_DSET
RES_DIR: Path = TWITCH_DSET / 'robots-views'

t1 = time.time()

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
    def __init__(self, y_min, y_max, threshold, img_pixels, image_mask):

        self.y_min = y_min
        self.y_max = y_max
        self.pixels = [
            (pix[0], pix[1])
            for pix in img_pixels
            if 20 < pix[1] < 70
        ]

        self.moyenne_R = self.get_moyenne(R, image_mask)
        self.moyenne_G = self.get_moyenne(G, image_mask)
        self.moyenne_B = self.get_moyenne(B, image_mask)
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
    if math.sqrt(pow(moy_r - zone.moyenne_R, 2) + pow(moy_g - zone.moyenne_G, 2) + pow(moy_b - zone.moyenne_B, 2)) < zone.threshold:
        return 1
    else:
        return 0


def is_image_from_robot_view(frame):
    return process_frame_moyennes(frame, mask_img)


# Opens the Video file
mask_img = Mask(f'{__file__}/../mask.jpg')
video_path = TWITCH_DSET / 'videos' / 'Rm.mp4'
cap = cv2.VideoCapture(str(video_path))

i = 0
while cap.isOpened():
    ret, rframe = cap.read()
    if not ret:
        break
    if i % 15 == 0:
        if process_frame_moyennes(rframe, mask_img):
            cv2.imwrite(f"{RES_DIR}/NewVisionAlone/-frame-{i+1:06}.jpg", rframe)
    i += 1


# t1 = time.time()
# for i in range (0, 10000):
#     processframemoyennes(frame, mask)       # 7.43s 10.17s
# print("FinishedFM", time.time() - t1)
#
# t1 = time.time()
# for i in range (0 , 10000):
#     processframe(frame, mask)                 # 11.26s ou 15.2s
# print("FinishedF", time.time() - t1)

# t1 = time.time()
# for i in range (0 , 10000):                  # 41.54s
#     ret, frame = cap.read()
# print("FinishedCap", time.time() - t1)

print("Finished New Vision Alone", time.time() - t1)
cap.release()
cv2.destroyAllWindows()


