from dataclasses import dataclass
from math import pi

from dynaconf import LazySettings
from injector import Module, provider, singleton, multiprovider

from object_detection.utils import label_map_util
from polystar.common.constants import LABEL_MAP_PATH
from polystar.common.models.camera import Camera
from polystar.common.utils.tensorflow import LabelMap


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
        return label_map_util.create_category_index_from_labelmap(str(LABEL_MAP_PATH), use_display_name=True)
