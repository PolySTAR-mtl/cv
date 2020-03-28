import random
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom.minidom import parseString

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import rotate_bound

# Facteur de redimensionnement. La taille de l'image sera divisée ou multipliée par ce chiffre au maximum
FACT_RESIZE = 1

# Max angle in degrees for random rotation
MAX_ANGLE = 20


def preprocess_background(path_back):
    background = cv2.imread(path_back)
    return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)


def preprocess_sticker(path_item):
    item = cv2.imread(path_item, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(item, cv2.COLOR_BGRA2RGBA)


def generate_one(
    background, item, rotate=True, scale=True, custom_rotate=None, custom_scale=None, to_print=False, save_name=None
):
    if rotate:
        if custom_rotate is None:
            percent_rotate = np.round((2 * random.random()) - 1, 3)  # entre -1 et 1
        else:
            percent_rotate = custom_rotate
        item = rotate_percentage(item, percent_rotate)
    if scale:
        if custom_scale is None:
            percent_scale = np.round((2 * random.random()) - 1, 3)  # entre -1 et 1
        else:
            percent_scale = custom_scale
        item = reshape_percentage(item, percent_scale)
    mask_alpha = item[:, :, 3]
    item = cv2.cvtColor(item, cv2.COLOR_RGBA2RGB)
    hs, ws, _ = item.shape

    background_subset, h_start, w_start = get_subset_shapes(background, hs, ws)
    composition_subset = superimpose(background_subset, item, mask_alpha)

    composition = background.copy()
    composition[h_start : h_start + hs, w_start : w_start + ws, :] = composition_subset

    labels = [h_start, w_start, h_start + hs, w_start + ws]

    if not (save_name is None):
        cv2.imwrite(save_name, cv2.cvtColor(composition, cv2.COLOR_RGB2BGR))

    if to_print:
        plt.imshow(composition)
        plt.show()

    return composition, labels


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
    h_start = int(random.random() * delta_h)
    w_start = int(random.random() * delta_w)
    return img_extract[h_start : h_start + hs, w_start : w_start + ws, :], h_start, w_start


def reshape_percentage(img_base, percent):
    intensity = 1 + abs(percent) * FACT_RESIZE
    h, w, _ = img_base.shape
    if percent < 0:
        h_dest, w_dest = int(h / intensity), int(w / intensity)
    else:
        h_dest, w_dest = h * int(intensity), w * int(intensity)

    return cv2.resize(img_base, (w_dest, h_dest), interpolation=cv2.INTER_AREA)


def rotate_percentage(img_base, percent):
    angle = MAX_ANGLE * percent
    return rotate_bound(img_base, angle)


path_background = "images_sup/back1.jpg"
path_sticker = "images_sup/logo.png"

background = preprocess_background(path_background)
sticker = preprocess_sticker(path_sticker)

labels = []
filenames = []
for i in range(10):
    folder = "dataset/"
    filename = "image_" + str(i) + ".jpg"
    filenames.append(filename)
    _, label = generate_one(background, sticker, to_print=False, save_name=folder + filename)
    labels.append(label)


data = ET.Element("annotations")

for i, [xmin, ymin, xmax, ymax] in enumerate(labels):
    object = ET.SubElement(data, "object")
    sub_name = ET.SubElement(object, "filename")
    sub_xmin = ET.SubElement(object, "xmin")
    sub_ymin = ET.SubElement(object, "ymin")
    sub_xmax = ET.SubElement(object, "xmax")
    sub_ymax = ET.SubElement(object, "ymax")
    sub_name.text = filenames[i]
    sub_xmin.text = str(xmin)
    sub_ymin.text = str(ymin)
    sub_xmax.text = str(xmax)
    sub_ymax.text = str(ymax)

dom = parseString(ET.tostring(data).decode("utf-8"))
pretty_xml = dom.toprettyxml()
myfile = Path("dataset/labels.xml")
myfile.write_text(pretty_xml)

# with open('dataset/labels.txt', 'w') as f:
#     for label in labels:
#         f.write("%s\n" % label)
