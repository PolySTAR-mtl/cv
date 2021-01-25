from shutil import move

from skimage import io
from tqdm import tqdm

from research.common.constants import TWITCH_DSET_DIR, TWITCH_ROBOTS_VIEWS_DIR
from research.common.dataset.twitch.aerial_view_detector import aerial_view_detector

AERIAL_VIEWS_DIR = TWITCH_DSET_DIR / "aerial-views"

AERIAL_VIEWS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    n = 0
    for file_path in tqdm(list(TWITCH_ROBOTS_VIEWS_DIR.glob("*.jpg")), unit="image", desc="Moving aerial views"):
        if aerial_view_detector.is_matching(io.imread(str(file_path))):
            move(str(file_path), str(AERIAL_VIEWS_DIR / file_path.name))
            n += 1

    print(f"Moved {n} images")