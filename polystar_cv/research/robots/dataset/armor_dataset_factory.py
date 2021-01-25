from itertools import islice
from typing import Iterator, List, Tuple

import matplotlib.pyplot as plt

from polystar.common.models.image import Image
from polystar.common.models.object import Armor, ObjectType
from polystar.common.target_pipeline.objects_validators.type_object_validator import TypeObjectValidator
from research.common.datasets.lazy_dataset import LazyDataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset import LazyROCODataset
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


class ArmorDataset(LazyDataset[Image, Armor]):
    def __init__(self, dataset: LazyROCODataset):
        super().__init__(f"{dataset.name}_armors")
        self.roco_dataset = dataset

    def __iter__(self) -> Iterator[Tuple[Image, Armor, str]]:
        for image, annotation, name in self.roco_dataset:
            yield from self._generate_from_single(image, annotation, name)

    @staticmethod
    def _generate_from_single(image: Image, annotation: ROCOAnnotation, name) -> Iterator[Tuple[Image, Armor, str]]:
        armors: List[Armor] = TypeObjectValidator(ObjectType.ARMOR).filter(annotation.objects, image)

        for i, obj in enumerate(armors):
            croped_img = image[obj.box.y1 : obj.box.y2, obj.box.x1 : obj.box.x2]
            yield croped_img, obj, f"{name}-{i}"


if __name__ == "__main__":
    for _armor_img, _armor, _name in islice(ArmorDataset(ROCODatasetsZoo.DJI.CENTRAL_CHINA.to_images()), 20, 30):
        print(_name, repr(_armor))
        plt.imshow(_armor_img)
        plt.show()
        plt.clf()
