from pathlib import Path

from polystar.constants import PROJECT_DIR
from polystar.settings import settings
from polystar.utils.path import make_path

DSET_DIR: Path = PROJECT_DIR / "dataset"

TWITCH_DSET_DIR = make_path(DSET_DIR / "twitch")
DJI_ROCO_DSET_DIR = make_path(DSET_DIR / "dji_roco")
DJI_ROCO_ZOOMED_DSET_DIR = make_path(DSET_DIR / "dji_roco_zoomed_v2")

TWITCH_ROBOTS_VIEWS_DIR = make_path(TWITCH_DSET_DIR / "robots-views")

IO_DIR = Path(settings.DRIVE_PATH) if settings.is_colab else PROJECT_DIR

EVALUATION_DIR = make_path(IO_DIR / "experiments")
TENSORFLOW_RECORDS_DIR = make_path(IO_DIR / "dataset/tf_records")
PIPELINES_DIR = make_path(IO_DIR / "pipelines")
