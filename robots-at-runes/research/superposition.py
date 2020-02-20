import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

threshold = 100


def superimpose(img1, img2, mask) :
    """
    Superimpose two pictures based on a mask. The two pictures
    must have the same shape.
    :param img1: first picture
    :param img2: second picture
    :param mask: mask applied on both pictures.
    'True' values will be applied on first picture
    :return: Composition
    """
    dst_shape = img1.shape
    composition = np.\
        zeros(dst_shape, dtype=np.uint8)
    for i in range(dst_shape[2]) :
        composition[:,:,i] = mask * img1[:,:,i] + ~mask * img2[:,:,i]
    return composition


background = cv2.imread("images_sup/back1.jpg")
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

logo = cv2.imread("images_sup/logo.png", cv2.IMREAD_UNCHANGED)
logo = cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA)

h,w,_ = background.shape
logo = cv2.resize(logo, (w,h), interpolation = cv2.INTER_AREA)

mask_alpha = (logo[:,:,3] < threshold)

logo = cv2.cvtColor(logo, cv2.COLOR_RGBA2RGB)

# logo = cv2.imread("images_sup/logo.png")
# logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)

composition = superimpose(background, logo, mask_alpha)

plt.imshow(background)
plt.show()
plt.imshow(logo)
plt.show()
plt.imshow(composition)
plt.show()