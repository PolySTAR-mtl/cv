from dataclasses import dataclass
from threading import Condition
from typing import Iterable, List, Tuple, Type

from injector import inject

from polystar.models.box import Box
from polystar.models.image import Image
from polystar.models.label_map import LabelMap
from polystar.models.roco_object import ObjectType
from polystar.settings import settings
from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.target_pipeline.detected_objects.objects_params import ObjectParams
from polystar.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC
from polystar.target_pipeline.objects_linker.objects_linker_abs import ObjectsLinkerABC
from polystar.utils.thread import MyThread
from polystar.utils.time import time_it
from research.common.constants import PIPELINES_DIR


@inject
@dataclass
class RobotsDetector:
    label_map: LabelMap
    object_detector_class: Type[ObjectsDetectorABC]
    objects_linker: ObjectsLinkerABC

    def flow_robots(self, images: Iterable[Image]) -> Iterable[List[DetectedRobot]]:
        detector_thread = ObjectsDetectorThread(self.object_detector_class, images)
        detector_thread.start()
        while detector_thread.running:
            image, objects_params = detector_thread.get_next_detection()
            robots, armors = self.make_robots_and_armors(objects_params, image)
            yield image, list(self.objects_linker.link_armors_to_robots(robots, armors, image))

    def make_robots_and_armors(
        self, objects_params: List[ObjectParams], image: Image
    ) -> Tuple[List[DetectedRobot], List[DetectedArmor]]:
        image_height, image_width, *_ = image.shape

        robots, armors = [], []
        for object_params in objects_params:
            object_type = ObjectType(self.label_map.name_of(object_params.object_class_id))
            box = Box.from_positions(
                # TODO what about using relative coordinates ?
                x1=int(object_params.xmin * image_width),
                y1=int(object_params.ymin * image_height),
                x2=int(object_params.xmax * image_width),
                y2=int(object_params.ymax * image_height),
            )
            if object_type is ObjectType.ARMOR:
                armors.append(DetectedArmor(object_type, box, object_params.score))
            else:
                robots.append(DetectedRobot(object_type, box, object_params.score))
        return robots, armors


class ObjectsDetectorThread(MyThread):
    def __init__(self, objects_detector_class: Type[ObjectsDetectorABC], images: Iterable[Image]):
        super().__init__()
        self.objects_detector_class = objects_detector_class
        self.images = images
        self.condition = Condition()
        self.image = None

    def get_next_detection(self) -> Tuple[Image, List[ObjectParams]]:
        with self.condition:
            self.condition.wait()
            return self.image, self.objects_params

    def loop(self):
        model_dir = PIPELINES_DIR / "roco-detection" / settings.OBJECTS_DETECTION_MODEL
        objects_detector = self.objects_detector_class(model_dir)
        for image in self.images:
            if not (image != self.image).any():
                continue

            objects_params = objects_detector.detect(image)

            with self.condition:
                self.image, self.objects_params = image, objects_params
                self.condition.notify()

            if not self.running:
                return
