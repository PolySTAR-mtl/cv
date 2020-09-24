from os import remove
from shutil import copy, make_archive, move, rmtree

from research.common.constants import TWITCH_DSET_DIR, TWITCH_DSET_ROBOTS_VIEWS_DIR, TWITCH_ROBOTS_VIEWS_DIR
from research.common.datasets_v3.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets_v3.roco.roco_dataset_descriptor import make_markdown_dataset_report
from research.common.scripts.construct_dataset_from_manual_annotation import construct_dataset_from_manual_annotations
from research.common.scripts.correct_annotations import AnnotationFileCorrector


def _construct_mixed_twitch_dset():
    construct_dataset_from_manual_annotations(
        TWITCH_ROBOTS_VIEWS_DIR, TWITCH_DSET_DIR / "reviewed-robots-views-annotations", TWITCH_DSET_ROBOTS_VIEWS_DIR,
    )


def _correct_manual_annotations():
    corrector = AnnotationFileCorrector(save_before=False)
    corrector.correct_annotations_in_directory(TWITCH_DSET_ROBOTS_VIEWS_DIR / "image_annotation")


def _extract_runes_images():
    all_twitch_dataset = _get_mixed_dataset_builder()
    for image_file, annotation, _ in all_twitch_dataset:
        if annotation.has_rune:
            copy(str(image_file), str(TWITCH_DSET_DIR / "runes" / image_file.name))


def _separate_twitch_videos():
    all_twitch_dataset_builder = _get_mixed_dataset_builder()
    for image_file, annotation, _ in all_twitch_dataset_builder:
        video_name = image_file.name.split("-")[0]
        dset_path = TWITCH_DSET_ROBOTS_VIEWS_DIR / video_name
        images_path = dset_path / "image"
        annotations_path = dset_path / "image_annotation"
        images_path.mkdir(exist_ok=True, parents=True)
        annotations_path.mkdir(exist_ok=True, parents=True)
        move(str(image_file), str(images_path / image_file.name))
        xml_name = f"{image_file.stem}.xml"
        move(str(all_twitch_dataset_builder.annotations_dir / xml_name), str(annotations_path / xml_name))
    if list((TWITCH_DSET_ROBOTS_VIEWS_DIR / "image").glob("*")):
        raise Exception(f"Some images remains unmoved")
    for remaining_file in (TWITCH_DSET_ROBOTS_VIEWS_DIR / "image_annotation").glob("*"):
        if remaining_file.name != ".DS_Store":
            raise Exception(f"Some annotations remains unmoved")
    rmtree(str(TWITCH_DSET_ROBOTS_VIEWS_DIR / "image"))
    rmtree(str(TWITCH_DSET_ROBOTS_VIEWS_DIR / "image_annotation"))


def _make_global_report():
    all_twitch_dataset_builder = _get_mixed_dataset_builder()
    make_markdown_dataset_report(all_twitch_dataset_builder.build_lazy(), all_twitch_dataset_builder.main_dir)


def _get_mixed_dataset_builder() -> ROCODatasetBuilder:
    return ROCODatasetBuilder(TWITCH_DSET_ROBOTS_VIEWS_DIR, "Twitch")


def _make_separate_reports():
    for video_dset_path in TWITCH_DSET_ROBOTS_VIEWS_DIR.glob("*"):
        if video_dset_path.is_dir():
            twitch_dset = ROCODatasetBuilder(video_dset_path, f"TWITCH_{video_dset_path.name}")
            make_markdown_dataset_report(twitch_dset.build_lazy(), twitch_dset.main_dir)


if __name__ == "__main__":

    for zip_file in (TWITCH_DSET_DIR / "reviewed-robots-views-annotations").glob("*.zip"):
        remove(str(zip_file))
    for chunk_dir in (TWITCH_DSET_DIR / "reviewed-robots-views-annotations").glob("chunk_*"):
        make_archive(chunk_dir, "zip", chunk_dir)

    _construct_mixed_twitch_dset()
    _correct_manual_annotations()
    _extract_runes_images()
    _make_global_report()
    _separate_twitch_videos()
    _make_separate_reports()
