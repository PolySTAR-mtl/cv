from itertools import chain

from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.dji.dji_roco_zoomed_datasets import DJIROCOZoomedDataset
from research_common.dataset.tensorflow_record import TensorflowRecordFactory
from research_common.dataset.twitch.twitch_roco_datasets import TwitchROCODataset
from research_common.dataset.union_dataset import UnionDataset


def create_one_record_per_roco_dset():
    for roco_set in chain(DJIROCODataset, DJIROCOZoomedDataset, TwitchROCODataset):
        TensorflowRecordFactory.from_dataset(roco_set)


if __name__ == "__main__":
    # create_one_record_per_roco_dset()

    TensorflowRecordFactory.from_dataset(
        UnionDataset(
            TwitchROCODataset.TWITCH_470149568,
            TwitchROCODataset.TWITCH_470150052,
            TwitchROCODataset.TWITCH_470151286,
            TwitchROCODataset.TWITCH_470152289,
            TwitchROCODataset.TWITCH_470152730,
        )
    )

    TensorflowRecordFactory.from_dataset(
        UnionDataset(
            TwitchROCODataset.TWITCH_470152838, TwitchROCODataset.TWITCH_470153081, TwitchROCODataset.TWITCH_470158483,
        )
    )

    TensorflowRecordFactory.from_dataset(
        UnionDataset(DJIROCODataset.CentralChina, DJIROCODataset.NorthChina, DJIROCODataset.SouthChina)
    )
    TensorflowRecordFactory.from_dataset(DJIROCODataset.Final)

    TensorflowRecordFactory.from_dataset(
        UnionDataset(
            DJIROCOZoomedDataset.CentralChina, DJIROCOZoomedDataset.NorthChina, DJIROCOZoomedDataset.SouthChina
        )
    )
    TensorflowRecordFactory.from_dataset(DJIROCOZoomedDataset.Final)
