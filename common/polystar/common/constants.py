from pathlib import Path

PROJECT_DIR: Path = Path(__file__).parent.parent.parent.parent
RESOURCES_DIR: Path = PROJECT_DIR / "resources"
MODELS_DIR: Path = RESOURCES_DIR / "models"

LABEL_MAP_PATH: Path = PROJECT_DIR / "dataset" / "tf_records" / "label_map.pbtxt"
