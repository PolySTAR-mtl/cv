from shutil import copy, move, rmtree

from research_common.constants import TWITCH_DSET_DIR, TWITCH_ROBOTS_VIEWS_DIR, TWITCH_DSET_ROBOTS_VIEWS_DIR
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset
from research_common.dataset.roco_dataset_descriptor import make_markdown_dataset_report
from research_common.scripts.construct_dataset_from_manual_annotation import construct_dataset_from_manual_annotations
from research_common.scripts.correct_annotations import AnnotationFileCorrector


def _construct_mixed_twitch_dset():
    construct_dataset_from_manual_annotations(
        TWITCH_ROBOTS_VIEWS_DIR, TWITCH_DSET_DIR / "robots-views-annotations", TWITCH_DSET_ROBOTS_VIEWS_DIR,
    )


def _correct_manual_annotations():
    corrector = AnnotationFileCorrector(save_before=False)
    corrector.correct_annotations_in_directory(TWITCH_DSET_ROBOTS_VIEWS_DIR / "image_annotation")


def _extract_runes_images():
    all_twitch_dataset = _get_mixed_dataset()
    for annotation in all_twitch_dataset.image_annotations:
        if annotation.has_rune:
            copy(str(annotation.image_path), str(TWITCH_DSET_DIR / "runes" / annotation.image_path.name))


def _separate_twitch_videos():
    all_twitch_dataset = _get_mixed_dataset()
    for annotation in all_twitch_dataset.image_annotations:
        video_name = annotation.image_path.name.split("-")[0]
        dset_path = TWITCH_DSET_ROBOTS_VIEWS_DIR / video_name
        images_path = dset_path / "image"
        annotations_path = dset_path / "image_annotation"
        images_path.mkdir(exist_ok=True, parents=True)
        annotations_path.mkdir(exist_ok=True, parents=True)
        move(str(annotation.image_path), str(images_path / annotation.image_path.name))
        move(str(annotation.xml_path), str(annotations_path / annotation.xml_path.name))
    if list((TWITCH_DSET_ROBOTS_VIEWS_DIR / "image").glob("*")):
        raise Exception(f"Some images remains unmoved")
    if list((TWITCH_DSET_ROBOTS_VIEWS_DIR / "image_annotation").glob("*")):
        raise Exception(f"Some annotations remains unmoved")
    rmtree(str(TWITCH_DSET_ROBOTS_VIEWS_DIR / "image"))
    rmtree(str(TWITCH_DSET_ROBOTS_VIEWS_DIR / "image_annotation"))


def _make_global_report():
    all_twitch_dataset = _get_mixed_dataset()
    make_markdown_dataset_report(all_twitch_dataset, all_twitch_dataset.dataset_path)


def _get_mixed_dataset():
    return DirectoryROCODataset(TWITCH_DSET_ROBOTS_VIEWS_DIR, "Twitch")


def _make_separate_reports():
    for video_dset_path in TWITCH_DSET_ROBOTS_VIEWS_DIR.glob("*"):
        if video_dset_path.is_dir():
            twitch_dset = DirectoryROCODataset(video_dset_path, f"TWITCH_{video_dset_path.name}")
            make_markdown_dataset_report(twitch_dset, twitch_dset.dataset_path)


if __name__ == "__main__":
    _construct_mixed_twitch_dset()
    _correct_manual_annotations()
    _extract_runes_images()
    _make_global_report()
    _separate_twitch_videos()
    _make_separate_reports()
