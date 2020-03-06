from enum import Enum
from pathlib import Path

from research_common.dataset.roco_dataset import ROCODataset


class Split(Enum):
    Val = "val"
    Train = "train"
    Test = "test"
    TrainVal = "trainval"

    def get_split_file(self, dataset: ROCODataset) -> Path:
        return (dataset.dataset_path / self.value).with_suffix(".txt")
