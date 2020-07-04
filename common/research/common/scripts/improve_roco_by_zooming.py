from tqdm import tqdm

from research.common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research.common.dataset.dji.dji_roco_zoomed_datasets import DJIROCOZoomedDataset
from research.common.dataset.improvement.zoom import Zoomer
from research.common.dataset.perturbations.image_modifiers.brightness import BrightnessModifier
from research.common.dataset.perturbations.image_modifiers.contrast import ContrastModifier
from research.common.dataset.perturbations.image_modifiers.saturation import SaturationModifier
from research.common.dataset.perturbations.perturbator import ImagePerturbator


def improve_dji_roco_dataset_by_zooming_and_perturbating(
    dset: DJIROCODataset, zoomer: Zoomer, perturbator: ImagePerturbator
):
    zoomed_dset: DJIROCOZoomedDataset = DJIROCOZoomedDataset[dset.name]
    zoomed_dset.dataset_path.mkdir(parents=True)

    for img in tqdm(dset.image_annotations, desc=f"Processing {dset}", unit="image", total=len(dset)):
        for i, zoomed_image in enumerate(zoomer.zoom(img), 1):
            zoomed_image._image = perturbator.perturbate(zoomed_image.image)
            zoomed_image.save_to_dir(zoomed_dset.dataset_path, f"{img.image_path.stem}_zoom_{i}")


def improve_all_dji_datasets_by_zooming_and_perturbating(zoomer: Zoomer, perturbator: ImagePerturbator):
    for _dset in DJIROCODataset:
        improve_dji_roco_dataset_by_zooming_and_perturbating(zoomer=zoomer, dset=_dset, perturbator=perturbator)


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
