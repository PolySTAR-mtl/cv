from typing import Type

from polystar.common.models.box import Box
from polystar.common.models.image import Image
from polystar.common.models.label_map import LabelMap
from polystar.common.models.object import ObjectType
from polystar.common.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot


class DetectedObjectFactory:
    def __init__(self, image: Image, label_map: LabelMap):
        self.label_map = label_map
        self.image_height, self.image_width, *_ = image.shape

    def from_relative_positions(
        self, xmin: float, ymin: float, xmax: float, ymax: float, score: float, object_class_id: int,
    ) -> DetectedObject:
        object_type = ObjectType(self.label_map.name_of(object_class_id))
        object_class = self._get_object_class_from_type(object_type)
        return object_class(
            type=object_type,
            confidence=score,
            box=Box.from_positions(
                x1=int(xmin * self.image_width),
                y1=int(ymin * self.image_height),
                x2=int(xmax * self.image_width),
                y2=int(ymax * self.image_height),
            ),
        )

    @staticmethod
    def _get_object_class_from_type(object_type: ObjectType) -> Type[DetectedObject]:
        return DetectedArmor if object_type is ObjectType.Armor else DetectedRobot
