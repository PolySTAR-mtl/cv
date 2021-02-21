import json
from os import remove
from pathlib import Path

from polystar.models.box import Box
from polystar.models.image import load_image
from polystar.utils.path import copy_file, move_file
from research.common.constants import TWITCH_DSET_DIR, TWITCH_ROBOTS_VIEWS_DIR
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.dataset.twitch.mask_detector import has_bonus_icon, is_aerial_view

AERIAL_DIR = TWITCH_DSET_DIR / "v2" / "aerial-views"
RUNES_DIR = TWITCH_DSET_DIR / "v2" / "runes"


def match_on_dataset(builder: ROCODatasetBuilder):
    twitch_id = builder.main_dir.name
    hd_images_directory = TWITCH_ROBOTS_VIEWS_DIR / twitch_id

    dataset_v2_directory = TWITCH_DSET_DIR / "v2" / twitch_id

    _move_images_with_720p_annotations(builder, dataset_v2_directory, hd_images_directory, twitch_id)
    _copy_changes_locks(builder, dataset_v2_directory)
    _move_aerials_views(hd_images_directory)


def _move_images_with_720p_annotations(
    builder: ROCODatasetBuilder, dataset_v2_directory: Path, hd_images_directory: Path, twitch_id: str
):
    dataset = builder.build_lazy()
    missing_images = []
    for image_file, annotation, _ in dataset:
        hd_image_file = hd_images_directory / image_file.name
        if hd_image_file.exists():
            hd_image = load_image(hd_image_file)
            if has_bonus_icon(hd_image):
                remove(hd_image_file)
                continue
            elif is_aerial_view(hd_image):
                directory = AERIAL_DIR
            elif annotation.has_rune:
                directory = RUNES_DIR
            else:
                directory = dataset_v2_directory
            move_file(hd_image_file, directory / "image")
            _scale_annotation(annotation, height=1080, width=1920)
            annotation.save_in_directory(directory / "image_annotation", image_file.stem)
        else:
            missing_images.append(str(image_file))
    print(f"{len(missing_images)} missing images in {twitch_id}")
    (hd_images_directory / "missing.json").write_text(json.dumps(missing_images))


def _scale_annotation(annotation: ROCOAnnotation, height: int, width: int):
    vertical_ratio, horizontal_ratio = height / annotation.h, width / annotation.w

    for obj in annotation.objects:
        obj.box = Box.from_positions(
            x1=int(obj.box.x1 * horizontal_ratio),
            y1=int(obj.box.y1 * vertical_ratio),
            x2=int(obj.box.x2 * horizontal_ratio),
            y2=int(obj.box.y2 * vertical_ratio),
        )

    annotation.w, annotation.h = width, height


def _copy_changes_locks(builder, dataset_v2_directory):
    for task in ("colors", "digits"):
        changes_lock = builder.main_dir / f"{task}/.changes"
        if changes_lock.exists():
            copy_file(changes_lock, dataset_v2_directory / task)


def _move_aerials_views(hd_images_directory):
    for hd_image_file in hd_images_directory.glob("*.jpg"):
        image = load_image(hd_image_file)
        if has_bonus_icon(image):
            remove(hd_image_file)
        elif is_aerial_view(image):
            move_file(hd_image_file, AERIAL_DIR / "unannotated_image")


if __name__ == "__main__":
    for _builder in ROCODatasetsZoo.TWITCH:
        match_on_dataset(_builder)
    for _new_twitch_id in ("470149066", "470152932"):
        _move_aerials_views(TWITCH_ROBOTS_VIEWS_DIR / _new_twitch_id)
