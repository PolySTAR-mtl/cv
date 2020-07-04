from tqdm import tqdm

from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.dji.dji_roco_zoomed_datasets import DJIROCOZoomedDataset
from research_common.dataset.improvement.zoom import Zoomer


def improve_dji_roco_dataset_by_zooming(dset: DJIROCODataset, zoomer: Zoomer):
    zoomed_dset: DJIROCOZoomedDataset = DJIROCOZoomedDataset[dset.name]
    zoomed_dset.dataset_path.mkdir(parents=True, exist_ok=True)

    for img in tqdm(dset.image_annotations, desc=f"Processing {dset}", unit="image", total=len(dset)):
        for i, zoomed_image in enumerate(zoomer.zoom(img), 1):
            zoomed_image.save_to_dir(zoomed_dset.dataset_path, f"{img.image_path.stem}_zoom_{i}")


def improve_all_dji_datasets_by_zooming(zoomer: Zoomer):
    for _dset in DJIROCODataset:
        improve_dji_roco_dataset_by_zooming(zoomer=zoomer, dset=_dset)


if __name__ == "__main__":
    improve_all_dji_datasets_by_zooming(Zoomer(854, 480, 0.15, 0.5))
