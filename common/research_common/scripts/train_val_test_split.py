from pathlib import Path
from typing import Iterable

from sklearn.model_selection import train_test_split

from research_common.dataset.directory_roco_dataset import DirectoryROCODataset
from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.split import Split


def _check_for_previous_split(dataset: DirectoryROCODataset):
    for split in Split:
        error_message = f"split {split.value}.txt already exists. Forbidden to overwrite for results consistency."
        assert not split.get_split_file(dataset).exists(), error_message


def _save_set_split(dataset: DirectoryROCODataset, split: Split, file_paths: Iterable[Path]):
    split.get_split_file(dataset).write_text("\n".join(file.stem for file in file_paths))


def _create_splits(dataset: DirectoryROCODataset):
    file_paths = list(dataset.image_paths)
    train_val_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=424242)
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=424242)

    _save_set_split(dataset, Split.Test, test_files)
    _save_set_split(dataset, Split.Train, train_files)
    _save_set_split(dataset, Split.Val, val_files)
    _save_set_split(dataset, Split.TrainVal, train_val_files)


def train_test_split_set(dataset: DirectoryROCODataset):
    _check_for_previous_split(dataset)
    _create_splits(dataset)


if __name__ == "__main__":
    for _roco_set in DJIROCODataset:
        train_test_split_set(_roco_set)
