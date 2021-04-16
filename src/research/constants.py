from pathlib import Path

from polystar.constants import PROJECT_DIR
from polystar.settings import settings
from polystar.utils.path import make_path

DSET_DIR: Path = PROJECT_DIR / "dataset"

TWITCH_DSET_DIR = make_path(DSET_DIR / "twitch")
DJI_ROCO_DSET_DIR = make_path(DSET_DIR / "dji_roco")
DJI_ROCO_ZOOMED_DSET_DIR = make_path(DSET_DIR / "dji_roco_zoomed_v2")

TWITCH_ROBOTS_VIEWS_DIR = make_path(TWITCH_DSET_DIR / "robots-views")

EVALUATION_DIR = make_path(PROJECT_DIR / "experiments")


if settings.is_colab:
    DRIVE_PATH = Path(settings.DRIVE_PATH)
    TENSORFLOW_RECORDS_DIR = make_path(DRIVE_PATH / "dataset/tf_records")
    PIPELINES_DIR = make_path(DRIVE_PATH / "pipelines")
else:
    TENSORFLOW_RECORDS_DIR = make_path(DSET_DIR / "tf_records")
    PIPELINES_DIR = make_path(PROJECT_DIR / "pipelines")
