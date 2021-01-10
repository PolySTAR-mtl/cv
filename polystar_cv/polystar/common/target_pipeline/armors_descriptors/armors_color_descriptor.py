from dataclasses import dataclass
from typing import List

from polystar.common.models.image import Image
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.target_pipeline.armors_descriptors.armors_descriptor_abc import ArmorsDescriptorABC
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor


@dataclass
class ArmorsColorDescriptor(ArmorsDescriptorABC):

    image_pipeline: ClassificationPipeline

    def _describe_armors_from_images(self, armors_images: List[Image], armors: List[DetectedArmor]):
        colors_predictions = self.image_pipeline.predict_proba(armors_images)
        for colors_proba, armor in zip(colors_predictions, armors):
            armor.colors_proba = colors_proba
