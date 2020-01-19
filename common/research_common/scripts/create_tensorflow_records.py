from research_common.dataset.roco.roco_datasets import ROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset
from research_common.dataset.tensorflow_record import create_tf_record_from_dataset, create_tf_record_from_datasets


def create_one_record_per_roco_dset():
    for roco_set in ROCODataset:
        for split in Split:
            create_tf_record_from_dataset(SplitDataset(roco_set, split))


def create_one_roco_record():
    for split in Split:
        create_tf_record_from_datasets(
            [SplitDataset(roco_dset, split) for roco_dset in ROCODataset], f"DJI_ROCO_{split.name}"
        )


if __name__ == "__main__":
    create_one_record_per_roco_dset()
    create_one_roco_record()
