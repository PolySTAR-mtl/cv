from enum import Enum
from pathlib import Path

from dynaconf import LazySettings

from polystar.common.constants import PROJECT_DIR


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(LazySettings):
    def get_env(self) -> Environment:
        return Environment(self.current_env.lower())


def _config_file_for_project(project_name: str) -> Path:
    return PROJECT_DIR / "config" / "settings.toml"


def make_settings() -> LazySettings:
    return LazySettings(
        SILENT_ERRORS_FOR_DYNACONF=False,
        SETTINGS_FILE_FOR_DYNACONF=f"{PROJECT_DIR  / 'config' / 'settings.toml'}",
        ENV_SWITCHER_FOR_DYNACONF="POLYSTAR_ENV",
    )


settings = make_settings()
