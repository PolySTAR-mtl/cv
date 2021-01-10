from dataclasses import dataclass

import numpy as np

from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC, ObjectT


@dataclass
class NegationValidator(ObjectsValidatorABC):
    validator: ObjectsValidatorABC

    def validate_single(self, obj: ObjectT, image: np.ndarray) -> bool:
        return not self.validator.validate_single(obj, image)
