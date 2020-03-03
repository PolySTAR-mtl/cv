from dataclasses import dataclass

import tensorflow as tf
from dynaconf import LazySettings
from injector import Injector, Module, provider, singleton
from tensorflow.compat.v1 import Session

from polystar.common.constants import MODELS_DIR
from polystar.common.dependency_injection import CommonModule
from polystar.common.models.tf_model import TFModel
from polystar.robots_at_robots.globals import settings


def make_injector() -> Injector:
    return Injector(modules=[CommonModule(settings), RobotsAtRobotsModule(settings)])


@dataclass
class RobotsAtRobotsModule(Module):
    settings: LazySettings

    @provider
    @singleton
    def provide_model(self) -> TFModel:
        print()
        model = tf.saved_model.load(export_dir=str(MODELS_DIR / self.settings.MODEL_NAME / "saved_model"))
        return model.signatures["serving_default"]
