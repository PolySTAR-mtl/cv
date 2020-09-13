from research.common.dataset.improvement.zoom import Zoomer
from research.common.dataset.perturbations.image_modifiers.brightness import \
    BrightnessModifier
from research.common.dataset.perturbations.image_modifiers.contrast import \
    ContrastModifier
from research.common.dataset.perturbations.image_modifiers.saturation import \
    SaturationModifier
from research.common.dataset.perturbations.perturbator import ImagePerturbator
from research.common.datasets.roco.roco_dataset import ROCOFileDataset
from research.common.datasets.roco.zoo.dji_zoomed import DJIROCOZoomedDatasets
from research.common.datasets.roco.zoo.roco_datasets_zoo import ROCODatasetsZoo
from tqdm import tqdm


def improve_dji_roco_dataset_by_zooming_and_perturbating(
    dset: ROCOFileDataset, zoomer: Zoomer, perturbator: ImagePerturbator
):
    zoomed_dset = DJIROCOZoomedDatasets.make_dataset(dset.name)
    zoomed_dset.create()

    for img, annotation in tqdm(dset.open(), desc=f"Processing {dset}", unit="image", total=len(dset)):
        for zoomed_image, zoomed_annotation in zoomer.zoom(img, annotation):
            zoomed_image = perturbator.perturbate(zoomed_image)
            zoomed_dset.add(zoomed_image, zoomed_annotation)


def improve_all_dji_datasets_by_zooming_and_perturbating(zoomer: Zoomer, perturbator: ImagePerturbator):
    for _dset in ROCODatasetsZoo.DJI:
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
