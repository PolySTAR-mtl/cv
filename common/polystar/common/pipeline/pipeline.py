from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from polystar.common.models.object import Object
from polystar.common.models.target_abc import TargetABC
from polystar.common.pipeline.object_selectors.object_selector_abc import ObjectSelectorABC
from polystar.common.pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC
from polystar.common.pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC
from polystar.common.pipeline.target_factories.target_factory_abc import TargetFactoryABC


class NoTargetFound(Exception):
    pass


def display_object(image: np.ndarray, obj: Object):
    plt.figure(figsize=(16, 12))
    plt.imshow(image)
    color = "red"
    rect = plt.Rectangle((obj.x, obj.y), obj.w, obj.h, linewidth=1, edgecolor=color, fill=False)
    print((obj.x, obj.y), obj.w, obj.h)
    plt.gca().add_patch(rect)
    plt.text(obj.x, obj.y - 2, f"{obj.type.name} ({int(obj.confidence * 100)}%)", color=color)
    plt.show()


@dataclass
class Pipeline:

    objects_detector: ObjectsDetectorABC
    objects_validators: List[ObjectsValidatorABC]
    object_selector: ObjectSelectorABC
    target_factory: TargetFactoryABC

    def process(self, image: np.ndarray) -> TargetABC:
        objects = self._get_objects_of_interest(image)
        target = self._get_best_target(image, objects)
        return target

    def _get_objects_of_interest(self, image: np.ndarray) -> List[Object]:
        objects = self.objects_detector.detect(image)
        for objects_validator in self.objects_validators:
            objects = objects_validator.filter(objects, image)

        if not objects:
            raise NoTargetFound()

        return objects

    def _get_best_target(self, image: np.ndarray, objects: List[Object]) -> TargetABC:
        selected_obj = self.object_selector.select(objects, image)
        display_object(image, selected_obj)
        target = self.target_factory.from_object(selected_obj, image)
        return target
