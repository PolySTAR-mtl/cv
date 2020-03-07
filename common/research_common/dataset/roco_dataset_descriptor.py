from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from pandas import DataFrame

from polystar.common.models.object import ObjectType, ArmorColor, ArmorNumber
from polystar.common.utils.markdown import MarkdownFile
from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.roco_dataset import ROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset
from research_common.dataset.twitch.twitch_roco_datasets import TwitchROCODataset


@dataclass
class ROCODatasetStats:
    n_images: int = 0

    n_runes: int = 0
    n_robots: int = 0
    n_watchers: int = 0
    n_bases: int = 0

    armors_color2num2count: Dict[str, Dict[int, int]] = field(default_factory=dict)

    @staticmethod
    def from_dataset(dataset: ROCODataset) -> ROCODatasetStats:
        rv = ROCODatasetStats()
        colors = ["red", "grey", "blue", "total"]
        rv.armors_color2num2count = {c: {n: 0 for n in range(10)} for c in colors}
        for c in colors:
            rv.armors_color2num2count[c]["total"] = 0
        for annotation in dataset.image_annotations:
            rv.n_images += 1
            rv.n_runes += annotation.has_rune
            for obj in annotation.objects:
                if obj.type == ObjectType.Car:
                    rv.n_robots += 1
                elif obj.type == ObjectType.Base:
                    rv.n_bases += 1
                elif obj.type == ObjectType.Watcher:
                    rv.n_watchers += 1
                elif obj.type == ObjectType.Armor:
                    rv.armors_color2num2count[obj.color.name.lower()][obj.numero] += 1
                    rv.armors_color2num2count[obj.color.name.lower()]["total"] += 1
                    rv.armors_color2num2count["total"][obj.numero] += 1
        return rv


def make_markdown_dataset_report(dataset: ROCODataset, report_dir: Path):
    report_path = report_dir / f"dset_{dataset.dataset_name}_report.md"

    stats = ROCODatasetStats.from_dataset(dataset)

    with MarkdownFile(report_path) as mf:
        mf.title(f"Dataset {dataset.dataset_name}")

        mf.paragraph(f"{stats.n_images} images, with:")
        mf.list(
            [
                f"{stats.n_bases} bases",
                f"{stats.n_watchers} watchers",
                f"{stats.n_runes} runes",
                f"{stats.n_robots} robots",
            ]
        )
        mf.table(DataFrame(stats.armors_color2num2count))


if __name__ == "__main__":
    for dset in TwitchROCODataset:
        make_markdown_dataset_report(dset, dset.dataset_path)
        # for split in Split:
        #     split_dset = SplitDataset(dset, split)
        #     make_markdown_dataset_report(split_dset, split_dset.dataset_path)

    for dset in DJIROCODataset:
        make_markdown_dataset_report(dset, dset.dataset_path)
        for split in Split:
            split_dset = SplitDataset(dset, split)
            make_markdown_dataset_report(split_dset, split_dset.dataset_path)
