from dataclasses import dataclass
from threading import Condition
from typing import Iterable, List, Tuple

from injector import inject

from polystar.models.box import Box
from polystar.models.image import Image
from polystar.models.label_map import LabelMap
from polystar.models.roco_object import ObjectType
from polystar.target_pipeline.detected_objects.detected_armor import DetectedArmor
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.target_pipeline.detected_objects.objects_params import ObjectParams
from polystar.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC
from polystar.target_pipeline.objects_linker.objects_linker_abs import ObjectsLinkerABC
from polystar.utils.thread import MyThread


@inject
@dataclass
class RobotsDetector:
    label_map: LabelMap
    object_detector: ObjectsDetectorABC
    objects_linker: ObjectsLinkerABC

    def flow_robots(self, images: Iterable[Image]) -> Iterable[List[DetectedRobot]]:
        detector_thread = ObjectsDetectorThread(self.object_detector, images)
        detector_thread.start()
        while detector_thread.running:
            with detector_thread.condition:
                detector_thread.condition.wait()  # TODO: timeout ?
                image, objects_params = detector_thread.image, detector_thread.objects_params
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
    def __init__(self, objects_detector: ObjectsDetectorABC, images: Iterable[Image]):
        super().__init__()
        self.objects_detector = objects_detector
        self.objects_params: List[ObjectParams] = None
        self.images = images
        self.condition = Condition()

    def loop(self):
        for image in self.images:
            objects_params = self.objects_detector.detect(image)

            with self.condition:
                self.image, self.objects_params = image, objects_params
                self.condition.notify()

            if not self.running:
                return
