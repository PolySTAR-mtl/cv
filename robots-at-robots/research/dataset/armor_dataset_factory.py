from typing import List, Iterable, Tuple

import matplotlib.pyplot as plt

from polystar.common.models.image import Image
from polystar.common.models.image_annotation import ImageAnnotation
from polystar.common.models.object import ObjectType, Armor, ArmorColor, ArmorNumber
from polystar.common.pipeline.objects_validators.type_object_validator import TypeObjectValidator
from research_common.dataset.dataset import Dataset
from research_common.dataset.roco.roco_datasets import ROCODataset


class ArmorDatasetFactory:
    @staticmethod
    def from_image_annotation(image_annotation: ImageAnnotation) -> Iterable[Tuple[Image, ArmorColor, int]]:
        img = image_annotation.image
        armors: List[Armor] = TypeObjectValidator(ObjectType.Armor).filter(image_annotation.objects, img)
        for obj in armors:
            yield img[obj.y : obj.y + obj.h, obj.x : obj.x + obj.w], obj.color, obj.numero

    @staticmethod
    def from_dataset(dataset: Dataset) -> Iterable[Tuple[Image, ArmorColor, ArmorNumber]]:
        for image_annotation in dataset.image_annotations:
            for rv in ArmorDatasetFactory.from_image_annotation(image_annotation):
                yield rv


if __name__ == "__main__":
    for i, (armor_img, c, n) in enumerate(ArmorDatasetFactory.from_dataset(ROCODataset.CentralChina)):
        print(c, n)
        plt.imshow(armor_img)
        plt.show()
        plt.clf()

        if i == 15:
            break
