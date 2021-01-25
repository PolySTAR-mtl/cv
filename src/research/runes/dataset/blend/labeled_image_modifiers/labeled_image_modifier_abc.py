from abc import ABC, abstractmethod
from random import random

from polystar.common.models.image import Image
from research.runes.dataset.labeled_image import LabeledImage, PointOfInterest


class LabeledImageModifierABC(ABC):
    def randomly_modify(self, labeled_image: LabeledImage) -> LabeledImage:
        return self.modify_from_factor(labeled_image, random() * 2 - 1)

    def modify_from_factor(self, labeled_image: LabeledImage, factor: float) -> LabeledImage:
        return self.modify_from_value(labeled_image, self._get_value_from_factor(factor))

    def modify_from_value(self, labeled_image: LabeledImage, value: float) -> LabeledImage:
        new_image = self._generate_modified_image(labeled_image.image, value)
        return LabeledImage(
            image=new_image,
            point_of_interests=[
                self._generate_modified_poi(poi, labeled_image.image, new_image, value)
                for poi in labeled_image.point_of_interests
            ],
        )

    @abstractmethod
    def _get_value_from_factor(self, factor: float) -> float:
        """
        :param factor: a factor of modification, in range [-1, 1]
        :return: the value of modification used by other function of this class
        """

    @abstractmethod
    def _generate_modified_image(self, image: Image, value: float) -> Image:
        pass

    @abstractmethod
    def _generate_modified_poi(
        self, poi: PointOfInterest, original_image: Image, new_image: Image, value: float
    ) -> PointOfInterest:
        pass
