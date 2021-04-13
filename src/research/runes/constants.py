from pathlib import Path

from research.constants import DSET_DIR

RUNES_DATASET_DIR: Path = DSET_DIR / "runes"


RUNES_DATASET_DIR.mkdir(exist_ok=True, parents=True)
