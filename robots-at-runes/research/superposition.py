import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random as rd
import constants as cst

percent_scale = np.round((2*rd.random())-1, 3)

def superimpose(img1, img2, mask):
    """
    Superimpose two pictures based on a mask. The two pictures
    must have the same shape.
    :param img1: first picture
    :param img2: second picture
    :param mask: mask applied on both pictures. Values between 0 and 255
    :return: Composition
    """
    dst_shape = img1.shape
    img1f = img1.astype(np.float)
    img2f = img2.astype(np.float)
    mix = np.zeros(dst_shape, dtype=np.uint8)
    for i in range(dst_shape[2]):
        mix[:, :, i] = ((~mask * img1f[:, :, i] + mask * img2f[:, :, i]) / 255).astype(np.uint8)
    return mix


def get_subset_shapes(img_extract, hs, ws):
    he, we, _ = img_extract.shape
    delta_h = he - hs
    delta_w = we - ws
    h_start = int(rd.random() * delta_h)
    w_start = int(rd.random() * delta_w)
    return img_extract[h_start:h_start + hs, w_start:w_start + ws, :], h_start, w_start


def reshape_percentage(img_base, percent):
    intensity = 1 + abs(percent) * cst.FACT_RESIZE
    h, w, _ = img_base.shape
    if percent < 0:
        h_dest, w_dest = int(h / intensity), int(w / intensity)
    else:
        h_dest, w_dest = h * int(intensity), w * int(intensity)

    return cv2.resize(img_base, (w_dest, h_dest), interpolation=cv2.INTER_AREA)


background = cv2.imread("images_sup/back1.jpg")
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

logo = cv2.imread("images_sup/logo.png", cv2.IMREAD_UNCHANGED)
logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA)
logo = reshape_percentage(logo, percent_scale)
mask_alpha = logo[:, :, 3]
logo = cv2.cvtColor(logo, cv2.COLOR_RGBA2RGB)
hs, ws, _ = logo.shape

background_subset, h_start, w_start = get_subset_shapes(background, hs, ws)
composition_subset = superimpose(background_subset, logo, mask_alpha)

composition = background.copy()
composition[h_start:h_start + hs, w_start:w_start + ws, :] = composition_subset

plt.imshow(background)
plt.show()
plt.imshow(logo)
plt.show()
plt.imshow(composition)
plt.show()
