from dataclasses import dataclass
from typing import List

from polystar.models.image import Image
from polystar.target_pipeline.armors_descriptors.armors_descriptor_abc import ArmorsDescriptorABC
from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from research.armors.armor_color.pipeline import ArmorColorPipeline


@dataclass
class ArmorsColorDescriptor(ArmorsDescriptorABC):

    image_pipeline: ArmorColorPipeline

    def __post_init__(self):
        assert ArmorColorPipeline.classes == self.image_pipeline.classes

    def _describe_armors_from_images(self, armors_images: List[Image], armors: List[DetectedArmor]):
        colors_predictions = self.image_pipeline.predict_proba(armors_images)
        for colors_proba, armor in zip(colors_predictions, armors):
            armor.colors_proba = colors_proba
