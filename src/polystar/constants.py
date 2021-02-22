from pathlib import Path

PROJECT_DIR: Path = Path(__file__).parent.parent.parent
RESOURCES_DIR: Path = PROJECT_DIR / "resources"

LABEL_MAP_PATH: Path = PROJECT_DIR / "dataset" / "label_map.pbtxt"

BYTE_ORDER = "little"
