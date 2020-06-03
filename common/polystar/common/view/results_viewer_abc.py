from abc import ABC, abstractmethod
from itertools import cycle
from typing import Iterable, NewType, Sequence, Tuple

from polystar.common.models.image import Image
from polystar.common.models.image_annotation import ImageAnnotation
from polystar.common.models.object import Object
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot, FakeDetectedRobot

ColorView = NewType("ColorView", Tuple[float, float, float])


class ResultViewerABC(ABC):
    def __init__(self, colors: Sequence[ColorView]):
        self.colors = colors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def new(self, image: Image):
        pass

    @abstractmethod
    def add_text(self, text: str, x: int, y: int, color: ColorView):
        pass

    @abstractmethod
    def add_rectangle(self, x: int, y: int, w: int, h: int, color: ColorView):
        pass

    @abstractmethod
    def display(self):
        pass

    def add_object(self, obj: Object, forced_color: ColorView = None):
        color = forced_color or self.colors[obj.type.value]
        self.add_rectangle(obj.box.x, obj.box.y, obj.box.w, obj.box.h, color)
        self.add_text(str(obj), obj.box.x, obj.box.y, color)

    def add_objects(self, objects: Iterable[Object], forced_color: ColorView = None):
        for obj in objects:
            self.add_object(obj, forced_color=forced_color)

    def display_image_with_objects(self, image: Image, objects: Iterable[Object]):
        self.new(image)
        self.add_objects(objects)
        self.display()

    def display_image_annotation(self, annotation: ImageAnnotation):
        self.display_image_with_objects(annotation.image, annotation.objects)

    def add_robot(self, robot: DetectedRobot, forced_color: ColorView = None):
        objects = robot.armors
        if not isinstance(robot, FakeDetectedRobot):
            objects = objects + [robot]

        self.add_objects(objects, forced_color)

    def add_robots(self, robots: Iterable[DetectedRobot], forced_color: ColorView = None):
        for color, robot in zip(cycle(self.colors), robots):
            self.add_robot(robot, forced_color=forced_color or color)
