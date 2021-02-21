from pathlib import Path

from imutils import resize

from polystar.models.image import load_image, save_image

save_image(resize(load_image(Path("mask_aerial.jpg")), 1920, 1080), Path("mask_aerial_red_hd.jpg"))
