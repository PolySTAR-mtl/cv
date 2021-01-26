from dataclasses import dataclass
from typing import List

from polystar.models.image import Image
from polystar.target_pipeline.armors_descriptors.armors_descriptor_abc import ArmorsDescriptorABC
from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from research.robots.armor_digit.pipeline import ArmorDigitPipeline


@dataclass
class ArmorsDigitDescriptor(ArmorsDescriptorABC):

    image_pipeline: ArmorDigitPipeline

    def __post_init__(self):
        assert ArmorDigitPipeline.classes == self.image_pipeline.classes

    def _describe_armors_from_images(self, armors_images: List[Image], armors: List[DetectedArmor]):
        digit_predictions = self.image_pipeline.predict_proba(armors_images)
        for digits_proba, armor in zip(digit_predictions, armors):
            armor.digits_proba = digits_proba
