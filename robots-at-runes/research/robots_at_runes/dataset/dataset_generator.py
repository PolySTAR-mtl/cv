import random
from pathlib import Path

import cv2
from tqdm import trange

from polystar.common.models.image import load_images_in_directory
from research.constants import RUNES_DATASET_DIR
from research.robots_at_runes.dataset import gaussian_noise, horizontal_blur
from research.robots_at_runes.dataset.blend import ImageBlender, LabeledImageRotator, LabeledImageScaler
from research.robots_at_runes.dataset.labeled_image import load_labeled_images_in_directory
from research.robots_at_runes.dataset.perturbations.perturbator import ImagePerturbator
from research.robots_at_runes.dataset.perturbations.perturbator_functions.contrast import contrast
from research.robots_at_runes.dataset.perturbations.perturbator_functions.gaussian_blur import gaussian_blur


class DatasetGenerator:
    def __init__(
        self, blender: ImageBlender, perturbator: ImagePerturbator, objects_directory: Path, backgrounds_directory: Path
    ):
        self.perturbator = perturbator
        self.blender = blender
        self.backgrounds = list(load_images_in_directory(backgrounds_directory, pattern="*.jpg"))
        self.objects = list(load_labeled_images_in_directory(objects_directory, cv2.COLOR_BGRA2RGBA, "png"))

    def generate(self, dataset_path: Path, n: int):
        dataset_path.mkdir(exist_ok=True, parents=True)

        for i in trange(n, desc="Generating runes dataset...", unit="img"):
            bg = random.choice(self.backgrounds)
            obj = random.choice(self.objects)

            labeled_image = self.blender.blend(bg, obj)
            labeled_image.image = self.perturbator.perturbate(labeled_image.image)

            labeled_image.save(dataset_path, f"img_{i:04d}")

    def generate_train_test(self, dataset_path: Path, n_train: int, n_test: int, n_val: int = 0):
        self.generate(dataset_path / "train", n_train)
        self.generate(dataset_path / "test", n_test)
        if n_val:
            self.generate(dataset_path / "val", n_val)


if __name__ == "__main__":
    generator = DatasetGenerator(
        ImageBlender(
            background_size=(1_280, 720), object_modifiers=[LabeledImageScaler(1.5), LabeledImageRotator(180)]
        ),
        ImagePerturbator([contrast, gaussian_blur, gaussian_noise, horizontal_blur]),
        RUNES_DATASET_DIR / "resources" / "objects",
        RUNES_DATASET_DIR / "resources" / "backgrounds",
    )

    generator.generate(RUNES_DATASET_DIR / "fake", 10)
