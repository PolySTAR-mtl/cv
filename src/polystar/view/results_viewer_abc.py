from abc import ABC, abstractmethod
from itertools import cycle
from typing import Iterable, Sequence, Tuple

from polystar.models.image import Image
from polystar.models.roco_object import ROCOObject
from polystar.target_pipeline.debug_pipeline import DebugInfo
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot, FakeDetectedRobot

ColorView = Tuple[float, float, float]


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

    def add_object(self, obj: ROCOObject, forced_color: ColorView = None):
        color = forced_color or self.colors[obj.type.value]
        self.add_rectangle(obj.box.x, obj.box.y, obj.box.w, obj.box.h, color)
        self.add_text(str(obj), obj.box.x, obj.box.y, color)

    def add_objects(self, objects: Iterable[ROCOObject], forced_color: ColorView = None):
        for obj in objects:
            self.add_object(obj, forced_color=forced_color)

    def display_image(self, image: Image):
        self.new(image)
        self.display()

    def display_image_with_objects(self, image: Image, objects: Iterable[ROCOObject]):
        self.new(image)
        self.add_objects(objects)
        self.display()

    def display_debug_info(self, debug_info: DebugInfo):
        self.add_debug_info(debug_info)
        self.display()

    def add_debug_info(self, debug_info: DebugInfo):
        self.new(debug_info.image)
        self.add_robots(debug_info.detected_robots, forced_color=(0.3, 0.3, 0.3))
        self.add_robots(debug_info.validated_robots)
        if debug_info.selected_armor is not None:
            self.add_object(debug_info.selected_armor)

    def add_robot(self, robot: DetectedRobot, forced_color: ColorView = None):
        objects = robot.armors
        if not isinstance(robot, FakeDetectedRobot):
            objects = objects + [robot]

        self.add_objects(objects, forced_color)

    def add_robots(self, robots: Iterable[DetectedRobot], forced_color: ColorView = None):
        for color, robot in zip(cycle(self.colors), robots):
            self.add_robot(robot, forced_color=forced_color or color)
