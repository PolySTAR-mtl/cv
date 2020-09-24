from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from pandas import DataFrame

from polystar.common.models.object import Armor, ObjectType
from polystar.common.utils.markdown import MarkdownFile
from polystar.common.utils.tqdm import smart_tqdm
from research.common.datasets_v3.roco.roco_dataset import LazyROCOFileDataset
from research.common.datasets_v3.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


@dataclass
class ROCODatasetStats:
    n_images: int = 0

    n_runes: int = 0
    n_robots: int = 0
    n_watchers: int = 0
    n_bases: int = 0

    armors_color2num2count: Dict[str, Dict[int, int]] = field(default_factory=dict)

    @staticmethod
    def from_dataset(dataset: LazyROCOFileDataset) -> "ROCODatasetStats":
        rv = ROCODatasetStats()
        colors = ["red", "grey", "blue", "total"]
        rv.armors_color2num2count = {c: {n: 0 for n in range(10)} for c in colors}
        for c in colors:
            rv.armors_color2num2count[c]["total"] = 0
        for (_, annotation, _) in smart_tqdm(dataset, desc=dataset.name, unit="frame"):
            rv.n_images += 1
            rv.n_runes += annotation.has_rune
            for obj in annotation.objects:
                if obj.type == ObjectType.Car:
                    rv.n_robots += 1
                elif obj.type == ObjectType.Base:
                    rv.n_bases += 1
                elif obj.type == ObjectType.Watcher:
                    rv.n_watchers += 1
                elif isinstance(obj, Armor):
                    rv.armors_color2num2count[obj.color.name.lower()][obj.number] += 1
                    rv.armors_color2num2count[obj.color.name.lower()]["total"] += 1
                    rv.armors_color2num2count["total"][obj.number] += 1
        return rv


def make_markdown_dataset_report(dataset: LazyROCOFileDataset, report_dir: Path):
    report_path = report_dir / f"dset_{dataset.name}_report.md"

    stats = ROCODatasetStats.from_dataset(dataset)

    with MarkdownFile(report_path) as mf:
        mf.title(f"Dataset {dataset.name}")

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
    dset = ROCODatasetsZoo.DJI.FINAL
    for datasets in ROCODatasetsZoo:
        make_markdown_dataset_report(datasets.union(), datasets.datasets_dir())
        for dset in datasets:
            make_markdown_dataset_report(dset.lazy_files(), dset.main_dir)
