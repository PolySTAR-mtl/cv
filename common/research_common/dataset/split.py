from enum import Enum
from pathlib import Path

from research_common.dataset.directory_roco_dataset import DirectoryROCODataset


class Split(Enum):
    Val = "val"
    Train = "train"
    Test = "test"
    TrainVal = "trainval"

    def get_split_file(self, dataset: DirectoryROCODataset) -> Path:
        return (dataset.dataset_path / self.value).with_suffix(".txt")
