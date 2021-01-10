from dataclasses import dataclass
from typing import List, Tuple, Type, Union

from polystar.common.models.box import Box
from polystar.common.models.image import Image
from polystar.common.models.label_map import LabelMap
from polystar.common.models.object import ObjectType
from polystar.common.target_pipeline.armors_descriptors.armors_descriptor_abc import ArmorsDescriptorABC
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot


@dataclass
class ObjectParams:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    score: float
    object_class_id: int


class DetectedObjectFactory:
    def __init__(self, label_map: LabelMap, armors_descriptors: List[ArmorsDescriptorABC]):
        self.armors_descriptors = armors_descriptors
        self.label_map = label_map
        self.image_height, self.image_width = None, None

    def make_lists(
        self, objects_params: List[ObjectParams], image: Image
    ) -> Tuple[List[DetectedRobot], List[DetectedArmor]]:
        self.image_height, self.image_width, *_ = image.shape
        robots, armors = [], []
        for object_params in objects_params:
            obj = self.from_object_params(object_params)
            if isinstance(obj, DetectedArmor):
                armors.append(obj)
            else:
                robots.append(obj)
        for armors_descriptor in self.armors_descriptors:
            armors_descriptor.describe_armors(image, armors)
        return robots, armors

    def from_object_params(self, object_params) -> Union[DetectedRobot, DetectedArmor]:
        object_type = ObjectType(self.label_map.name_of(object_params.object_class_id))
        object_class = self._get_object_class_from_type(object_type)
        return object_class(
            type=object_type,
            confidence=object_params.score,
            box=Box.from_positions(
                x1=int(object_params.xmin * self.image_width),
                y1=int(object_params.ymin * self.image_height),
                x2=int(object_params.xmax * self.image_width),
                y2=int(object_params.ymax * self.image_height),
            ),
        )

    @staticmethod
    def _get_object_class_from_type(object_type: ObjectType) -> Type[Union[DetectedRobot, DetectedArmor]]:
        return DetectedArmor if object_type is ObjectType.Armor else DetectedRobot
