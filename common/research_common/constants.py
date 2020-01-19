from pathlib import Path

DSET_DIR = Path(__file__).parent.parent.parent / "dataset"

TWITCH_DSET_DIR: Path = DSET_DIR / "twitch"
ROCO_DSET_DIR: Path = DSET_DIR / "dji_roco"
TENSORFLOW_RECORDS_DIR: Path = DSET_DIR / "tf_records"

TWITCH_DSET_DIR.mkdir(parents=True, exist_ok=True)
ROCO_DSET_DIR.mkdir(parents=True, exist_ok=True)
TENSORFLOW_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
