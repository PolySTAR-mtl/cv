from itertools import islice
from typing import Iterator, List, Tuple

import matplotlib.pyplot as plt
from polystar.common.models.image import Image
from polystar.common.models.object import Armor, ObjectType
from polystar.common.target_pipeline.objects_validators.type_object_validator import \
    TypeObjectValidator
from research.common.datasets.dataset import Dataset, GeneratorDataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset import (ROCODataset,
                                                        ROCOFileDataset)
from research.common.datasets.roco.zoo.roco_datasets_zoo import ROCODatasetsZoo

ArmorDataset = Dataset[Image, Armor]


class ArmorDatasetFactory:
    def __init__(self, dataset: ROCOFileDataset):
        self.dataset: ROCODataset = dataset.open()

    def make(self) -> ArmorDataset:
        return GeneratorDataset(f"{self.dataset.name}_armors", self._make_generator)

    def _make_generator(self) -> Iterator[Tuple[Image, Armor, str]]:
        for image, annotation, name in self.dataset:
            yield from self._generate_from_single(image, annotation, name)

    @staticmethod
    def _generate_from_single(image: Image, annotation: ROCOAnnotation, name) -> Iterator[Tuple[Image, Armor, str]]:
        armors: List[Armor] = TypeObjectValidator(ObjectType.Armor).filter(annotation.objects, image)

        for i, obj in enumerate(armors):
            croped_img = image[obj.box.y1 : obj.box.y2, obj.box.x1 : obj.box.x2]
            yield croped_img, obj, f"{name}-{i}"


if __name__ == "__main__":
    for _armor_img, _armor, _name in islice(ArmorDatasetFactory(ROCODatasetsZoo.DJI.CentralChina).make(), 20, 30):
        print(_name, repr(_armor))
        plt.imshow(_armor_img)
        plt.show()
        plt.clf()
