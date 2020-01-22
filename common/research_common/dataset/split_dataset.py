from pathlib import Path
from typing import Iterable, List

from research_common.dataset.dataset import Dataset
from research_common.dataset.roco.roco_datasets import ROCODataset
from research_common.dataset.split import Split


class SplitDataset(Dataset):
    def __init__(self, root_dataset: Dataset, split: Split):
        super().__init__(root_dataset.dataset_path, f"{root_dataset.dataset_name}_{split.name}")
        self._load_file_names(split)

    def _load_file_names(self, split: Split):
        self._file_names = split.get_split_file(self).read_text().strip().split("\n")

    @property
    def image_paths(self) -> Iterable[Path]:
        return self._generate_file_paths(self.images_dir_path, ".jpg")

    @property
    def annotation_paths(self) -> Iterable[Path]:
        return self._generate_file_paths(self.annotations_dir_path, ".xml")

    def _generate_file_paths(self, dir_path: Path, suffix: str) -> List[Path]:
        return [(dir_path / file_name).with_suffix(suffix) for file_name in self._file_names]


if __name__ == "__main__":
    print(
        set(SplitDataset(ROCODataset.CentralChina, Split.Train).annotation_paths)
        & set(SplitDataset(ROCODataset.CentralChina, Split.Val).annotation_paths)
    )
