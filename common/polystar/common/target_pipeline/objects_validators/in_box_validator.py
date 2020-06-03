from dataclasses import dataclass

from polystar.common.models.box import Box
from polystar.common.models.image import Image
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC, ObjectT


@dataclass
class InBoxValidator(ObjectsValidatorABC[ObjectT]):
    box: Box
    min_percentage_intersection: float

    def validate_single(self, obj: ObjectT, image: Image) -> bool:
        return self.box.contains(obj.box, self.min_percentage_intersection)
