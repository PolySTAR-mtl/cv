from itertools import islice

import matplotlib.pyplot as plt

from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.armor_digit.armor_digit_dataset import make_armor_digit_dataset_generator

if __name__ == "__main__":
    _roco_dataset = ROCODatasetsZoo.TWITCH.T470149568

    _armor_digit_dataset = make_armor_digit_dataset_generator().from_roco_dataset(_roco_dataset).to_file_images()

    for _img, _digit, _name in islice(_armor_digit_dataset, 10):
        print(_digit, _img.path)
        plt.imshow(_img)
        plt.show()
