from dataclasses import dataclass
from math import pi

from dynaconf import LazySettings
from injector import Module, provider, singleton, multiprovider, Injector

from polystar.common import settings
from polystar.common.constants import LABEL_MAP_PATH
from polystar.common.models.camera import Camera
from polystar.common.utils.tensorflow import LabelMap


def make_common_injector() -> Injector:
    return Injector(modules=[CommonModule(settings)])


@dataclass
class CommonModule(Module):
    settings: LazySettings

    @provider
    @singleton
    def provide_camera(self) -> Camera:
        return Camera(
            self.settings.CAMERA_HORIZONTAL_FOV / 180 * pi, self.settings.CAMERA_WIDTH, self.settings.CAMERA_HEIGHT
        )

    @multiprovider
    @singleton
    def provide_label_map(self) -> LabelMap:
        return LabelMap.from_file(LABEL_MAP_PATH)