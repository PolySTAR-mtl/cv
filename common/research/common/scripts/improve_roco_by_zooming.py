from pathlib import Path
from typing import Tuple

from polystar.common.models.image import save_image
from polystar.common.utils.str_utils import camel2snake
from polystar.common.utils.tqdm import smart_tqdm
from research.common.constants import DJI_ROCO_ZOOMED_DSET_DIR
from research.common.dataset.improvement.zoom import Zoomer
from research.common.dataset.perturbations.image_modifiers.brightness import BrightnessModifier
from research.common.dataset.perturbations.image_modifiers.contrast import ContrastModifier
from research.common.dataset.perturbations.image_modifiers.saturation import SaturationModifier
from research.common.dataset.perturbations.perturbator import ImagePerturbator
from research.common.datasets_v3.roco.roco_dataset import LazyROCODataset
from research.common.datasets_v3.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


def improve_dji_roco_dataset_by_zooming_and_perturbating(
    dset: LazyROCODataset, zoomer: Zoomer, perturbator: ImagePerturbator
):
    image_dir, annotation_dir = _prepare_empty_zoomed_dir(DJI_ROCO_ZOOMED_DSET_DIR / camel2snake(dset.name).lower())

    for img, annotation, name in smart_tqdm(dset, desc=f"Processing {dset}", unit="image"):
        for zoomed_image, zoomed_annotation, zoomed_name in zoomer.zoom(img, annotation, name):
            zoomed_image = perturbator.perturbate(zoomed_image)
            save_image(zoomed_image, image_dir / f"{zoomed_name}.jpg")
            (annotation_dir / f"{zoomed_name}.xml").write_text(zoomed_annotation.to_xml())


def improve_all_dji_datasets_by_zooming_and_perturbating(zoomer: Zoomer, perturbator: ImagePerturbator):
    for _dset in ROCODatasetsZoo.DJI:
        improve_dji_roco_dataset_by_zooming_and_perturbating(
            zoomer=zoomer, dset=_dset.to_images().build_lazy(), perturbator=perturbator
        )


def _prepare_empty_zoomed_dir(dir_path: Path) -> Tuple[Path, Path]:
    dir_path.mkdir()

    annotation_dir = dir_path / "image_annotation"
    image_dir = dir_path / "image"

    annotation_dir.mkdir()
    image_dir.mkdir()

    return image_dir, annotation_dir


if __name__ == "__main__":
    improve_all_dji_datasets_by_zooming_and_perturbating(
        Zoomer(w=854, h=480, max_overlap=0.15, min_coverage=0.5),
        ImagePerturbator(
            [
                ContrastModifier(min_coef=0.7, max_coef=1.5),
                BrightnessModifier(max_offset=10.0),
                SaturationModifier(max_saturation=0.6),
            ],
            min_intensity=0.5,
        ),
    )
