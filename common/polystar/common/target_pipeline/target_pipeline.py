from dataclasses import dataclass
from typing import List

import numpy as np

from polystar.common.communication.target_sender_abc import TargetSenderABC
from polystar.common.models.image import Image
from polystar.common.models.object import Object
from polystar.common.models.target_abc import TargetABC
from polystar.common.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC
from polystar.common.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC
from polystar.common.target_pipeline.target_factories.target_factory_abc import TargetFactoryABC


class NoTargetFound(Exception):
    pass


@dataclass
class TargetPipeline:

    objects_detector: ObjectsDetectorABC
    objects_validators: List[ObjectsValidatorABC]
    object_selector: ObjectSelectorABC
    target_factory: TargetFactoryABC
    target_sender: TargetSenderABC

    def predict_target(self, image: Image) -> TargetABC:
        selected_object = self.predict_best_object(image)
        target = self.target_factory.from_object(selected_object, image)
        self.target_sender.send(target)
        return target

    def predict_best_object(self, image: Image) -> Object:
        objects = self._get_objects_of_interest(image)
        selected_object = self.object_selector.select(objects, image)
        return selected_object

    def _get_objects_of_interest(self, image: np.ndarray) -> List[Object]:
        objects = self._detect_all_objects(image)
        for objects_validator in self.objects_validators:
            objects = objects_validator.filter(objects, image)

        if not objects:
            raise NoTargetFound()

        return objects

    def _detect_all_objects(self, image) -> List[Object]:
        return self.objects_detector.detect(image)
