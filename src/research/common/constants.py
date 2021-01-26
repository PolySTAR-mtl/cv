from pathlib import Path

from polystar.constants import PROJECT_DIR

DSET_DIR: Path = PROJECT_DIR / "dataset"

TWITCH_DSET_DIR: Path = DSET_DIR / "twitch"
DJI_ROCO_DSET_DIR: Path = DSET_DIR / "dji_roco"
DJI_ROCO_ZOOMED_DSET_DIR: Path = DSET_DIR / "dji_roco_zoomed_v2"
TENSORFLOW_RECORDS_DIR: Path = DSET_DIR / "tf_records"
TWITCH_ROBOTS_VIEWS_DIR: Path = TWITCH_DSET_DIR / "robots-views"
TWITCH_DSET_ROBOTS_VIEWS_DIR: Path = TWITCH_DSET_DIR / "final-robots-views"

TWITCH_DSET_DIR.mkdir(parents=True, exist_ok=True)
DJI_ROCO_DSET_DIR.mkdir(parents=True, exist_ok=True)
DJI_ROCO_ZOOMED_DSET_DIR.mkdir(parents=True, exist_ok=True)
TENSORFLOW_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
TWITCH_ROBOTS_VIEWS_DIR.mkdir(parents=True, exist_ok=True)
TWITCH_DSET_ROBOTS_VIEWS_DIR.mkdir(parents=True, exist_ok=True)


EVALUATION_DIR: Path = PROJECT_DIR / "experiments"
PIPELINES_DIR = PROJECT_DIR / "pipelines"
