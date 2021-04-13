from pathlib import Path

from tqdm import tqdm

from polystar.utils.path import copy_file
from research.common.datasets.roco.roco_annotation import ROCOAnnotation, move_image_and_annotation_from_directory
from research.constants import TWITCH_DSET_DIR
from research.dataset.twitch.annotation_file_corrector import AnnotationFileCorrector


def add_annotated_images_to_dataset(images_dir: Path, annotations_dir: Path, dataset_dir: Path):
    corrector = AnnotationFileCorrector(save_before=False)
    for annotation_file in tqdm(list(annotations_dir.glob("**/*.xml"))):
        sub_dataset_dir = dataset_dir / annotation_file.stem.split("-")[0]
        image_file = (images_dir / annotation_file.relative_to(annotations_dir)).with_suffix(".jpg")

        copy_file(image_file, sub_dataset_dir / "image")
        annotation_file = copy_file(annotation_file, sub_dataset_dir / "image_annotation")
        corrector.correct_annotation_file(annotation_file)

        if ROCOAnnotation.from_xml_file(annotation_file).has_rune:
            move_image_and_annotation_from_directory(sub_dataset_dir, dataset_dir / "runes", annotation_file.stem)


if __name__ == "__main__":
    add_annotated_images_to_dataset(
        TWITCH_DSET_DIR / "chunks-to-annotate", TWITCH_DSET_DIR / "chunks-annotations", TWITCH_DSET_DIR / "v2"
    )
