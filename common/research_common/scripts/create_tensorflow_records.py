from injector import inject

from polystar.common.dependency_injection import make_common_injector
from polystar.common.models.label_map import LabelMap
from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset
from research_common.dataset.tensorflow_record import TensorflowRecordFactory


@inject
def create_one_record_per_roco_dset(label_map: LabelMap):
    for roco_set in DJIROCODataset:
        for split in Split:
            TensorflowRecordFactory(label_map).from_dataset(SplitDataset(roco_set, split))


@inject
def create_one_roco_record(label_map: LabelMap):
    for split in Split:
        TensorflowRecordFactory(label_map).from_datasets(
            [SplitDataset(roco_dset, split) for roco_dset in DJIROCODataset], f"DJI_ROCO_{split.name}"
        )


if __name__ == "__main__":
    injector = make_common_injector()
    injector.call_with_injection(create_one_record_per_roco_dset)
    injector.call_with_injection(create_one_roco_record)
