import logging

from tqdm import tqdm

from research.common.dataset.cleaning.dataset_changes import DatasetChanges
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.roco_datasets import ROCODatasets
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.common.gcloud.gcloud_storage import GCStorages


def upload_all_digit_datasets(roco_datasets: ROCODatasets):
    for roco_dataset in tqdm(roco_datasets, desc="Uploading datasets"):
        upload_digit_dataset(roco_dataset)


def upload_all_color_datasets(roco_datasets: ROCODatasets):
    for roco_dataset in tqdm(roco_datasets, desc="Uploading datasets"):
        upload_color_dataset(roco_dataset)


def upload_digit_dataset(roco_dataset: ROCODatasetBuilder):
    _upload_armor_dataset(roco_dataset, "digits")


def upload_color_dataset(roco_dataset: ROCODatasetBuilder):
    _upload_armor_dataset(roco_dataset, "colors")


def _upload_armor_dataset(roco_dataset: ROCODatasetBuilder, name: str):
    GCStorages.DEV.upload_directory(roco_dataset.main_dir / name, extensions_to_exclude={".changes"})
    DatasetChanges(roco_dataset.main_dir / name).upload()


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    upload_all_digit_datasets(ROCODatasetsZoo.TWITCH)
    upload_digit_dataset(ROCODatasetsZoo.DJI.FINAL)
