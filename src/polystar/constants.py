from pathlib import Path

from polystar.settings import settings

PROJECT_DIR: Path = "/content/drive/My Drive/PolySTAR/3. RoboMaster/Équipe-Computer vision" if settings.PLATFORM == "colab" else Path(
    __file__
).parent.parent.parent
RESOURCES_DIR: Path = PROJECT_DIR / "resources"

LABEL_MAP_PATH: Path = PROJECT_DIR / "dataset" / "label_map.pbtxt"

BYTE_ORDER = "little"
