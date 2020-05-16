from pathlib import Path
from typing import List, Iterable, Tuple

import matplotlib.pyplot as plt

from polystar.common.models.image import Image
from polystar.common.models.image_annotation import ImageAnnotation
from polystar.common.models.object import ObjectType, Armor
from polystar.common.target_pipeline.objects_validators.type_object_validator import TypeObjectValidator
from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.roco_dataset import ROCODataset


class ArmorDatasetFactory:
    @staticmethod
    def from_image_annotation(image_annotation: ImageAnnotation) -> Iterable[Tuple[Image, Armor, int, Path]]:
        img = image_annotation.image
        armors: List[Armor] = TypeObjectValidator(ObjectType.Armor).filter(image_annotation.objects, img)
        for i, obj in enumerate(armors):
            croped_img = img[obj.box.y1 : obj.box.y2, obj.box.x1 : obj.box.x2]
            yield croped_img, obj, i, image_annotation.image_path

    @staticmethod
    def from_dataset(dataset: ROCODataset) -> Iterable[Tuple[Image, Armor, int, Path]]:
        for image_annotation in dataset.image_annotations:
            for rv in ArmorDatasetFactory.from_image_annotation(image_annotation):
                yield rv


if __name__ == "__main__":
    for i, (armor_img, armor, k, p) in enumerate(ArmorDatasetFactory.from_dataset(DJIROCODataset.CentralChina)):
        print(armor, armor.color, armor.number, "-", k, "in", p)
        plt.imshow(armor_img)
        plt.show()
        plt.clf()

        if i == 50:
            break
