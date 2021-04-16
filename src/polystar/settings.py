from enum import Enum

from dynaconf import LazySettings

from polystar.constants import PROJECT_DIR


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    COLAB = "colab"


class Settings(LazySettings):
    @property
    def env(self) -> Environment:
        return Environment(self.current_env.lower())

    @property
    def is_prod(self) -> bool:
        return self.env == Environment.PRODUCTION

    @property
    def is_dev(self) -> bool:
        return self.env == Environment.DEVELOPMENT

    @property
    def is_colab(self) -> bool:
        return self.env == Environment.COLAB


settings = Settings(
    SILENT_ERRORS_FOR_DYNACONF=False,
    SETTINGS_FILE_FOR_DYNACONF=f"{PROJECT_DIR / 'settings.toml'}",
    ENV_SWITCHER_FOR_DYNACONF="POLYSTAR_ENV",
)
