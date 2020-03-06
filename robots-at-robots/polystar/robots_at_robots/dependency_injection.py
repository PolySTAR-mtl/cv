from dataclasses import dataclass

from dynaconf import LazySettings
from injector import Injector, Module

from polystar.common.dependency_injection import CommonModule
from polystar.robots_at_robots.globals import settings


def make_injector() -> Injector:
    return Injector(modules=[CommonModule(settings), RobotsAtRobotsModule(settings)])


@dataclass
class RobotsAtRobotsModule(Module):
    settings: LazySettings
