from abc import ABC, abstractmethod
from typing import List

from polystar.common.models.image import Image
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor


class ArmorsDescriptorABC(ABC):
    def describe_armors(self, image: Image, armors: List[DetectedArmor]):
        armors_images = [image[armor.box.y1 : armor.box.y2, armor.box.x1 : armor.box.x2] for armor in armors]
        self._describe_armors_from_images(armors_images, armors)

    @abstractmethod
    def _describe_armors_from_images(self, armors_images: List[Image], armors: List[DetectedArmor]):
        pass
