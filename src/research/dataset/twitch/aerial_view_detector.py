from pathlib import Path

from skimage import io

from research.common.constants import TWITCH_DSET_DIR
from research.dataset.twitch.mask_detector import MaskDetector

aerial_view_detector = MaskDetector(
    Path(__file__).parent / "mask_aerial.jpg",
    [
        (527, 528, 292, 297, 20),
        (527, 531, 303, 303, 20),
        (532, 537, 286, 287, 20),
        (536, 541, 302, 303, 20),
        (543, 544, 292, 297, 20),
        (535, 535, 292, 297, 20),
    ],
)

if __name__ == "__main__":
    for file_path in sorted((TWITCH_DSET_DIR / "robots-views").glob("*.jpg")):
        if aerial_view_detector.is_matching(io.imread(str(file_path))):
            print(file_path.name)
