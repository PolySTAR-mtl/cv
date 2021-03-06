import random
from pathlib import Path

import cv2
from tqdm import trange

from polystar.models.image import load_images_in_directory
from research.dataset.perturbations.image_modifiers.brightness import BrightnessModifier
from research.dataset.perturbations.image_modifiers.contrast import ContrastModifier
from research.dataset.perturbations.image_modifiers.gaussian_blur import GaussianBlurrer
from research.dataset.perturbations.image_modifiers.gaussian_noise import GaussianNoiser
from research.dataset.perturbations.image_modifiers.horizontal_blur import HorizontalBlurrer
from research.dataset.perturbations.image_modifiers.saturation import SaturationModifier
from research.dataset.perturbations.perturbator import ImagePerturbator
from research.runes.constants import RUNES_DATASET_DIR
from research.runes.dataset.blend import LabeledImageScaler
from research.runes.dataset.blend.image_blender import ImageBlender
from research.runes.dataset.blend.labeled_image_modifiers.labeled_image_rotator import LabeledImageRotator
from research.runes.dataset.labeled_image import load_labeled_images_in_directory


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
        ImagePerturbator(
            [
                ContrastModifier(min_coef=0.7, max_coef=1.5),
                GaussianBlurrer(max_factor=0.015),
                GaussianNoiser(max_variance=300.0),
                HorizontalBlurrer(max_kernel_size=11),
                SaturationModifier(max_saturation=0.6),
                BrightnessModifier(max_offset=10.0),
            ]
        ),
        RUNES_DATASET_DIR / "resources" / "objects",
        RUNES_DATASET_DIR / "resources" / "backgrounds",
    )

    generator.generate(RUNES_DATASET_DIR / "fake", 10)
