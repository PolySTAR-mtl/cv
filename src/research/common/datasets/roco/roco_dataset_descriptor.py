from dataclasses import dataclass, field
from typing import Dict, Type

from pandas import DataFrame

from polystar.models.roco_object import Armor, ObjectType
from polystar.utils.dataframe import add_percentages_to_df
from polystar.utils.iterable_utils import apply
from polystar.utils.markdown import MarkdownFile
from polystar.utils.tqdm import smart_tqdm
from research.common.datasets.roco.roco_dataset import LazyROCOFileDataset
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.roco.roco_datasets import ROCODatasets
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


@dataclass
class ROCODatasetStats:
    n_images: int = 0

    n_runes: int = 0
    n_robots: int = 0
    n_watchers: int = 0
    n_bases: int = 0

    armors_df: DataFrame = field(default_factory=DataFrame)

    @property
    def numbers(self) -> Dict[str, int]:
        return {
            "images": self.n_images,
            # "runes": self.n_runes,
            "robots": self.n_robots,
            "watchers": self.n_watchers,
            "bases": self.n_bases,
        }

    @staticmethod
    def from_dataset(dataset: LazyROCOFileDataset) -> "ROCODatasetStats":
        rv = ROCODatasetStats()
        colors = ["red", "grey", "blue", "total"]
        armors_color2num2count = {c: {n: 0 for n in range(10)} for c in colors}
        for c in colors:
            armors_color2num2count[c]["total"] = 0
        for (_, annotation, _) in smart_tqdm(dataset, desc=dataset.name, unit="frame"):
            rv.n_images += 1
            rv.n_runes += annotation.has_rune
            for obj in annotation.objects:
                if obj.type == ObjectType.CAR:
                    rv.n_robots += 1
                elif obj.type == ObjectType.BASE:
                    rv.n_bases += 1
                elif obj.type == ObjectType.WATCHER:
                    rv.n_watchers += 1
                elif isinstance(obj, Armor):
                    armors_color2num2count[obj.color.name.lower()][obj.number] += 1
                    armors_color2num2count[obj.color.name.lower()]["total"] += 1
                    armors_color2num2count["total"][obj.number] += 1
        rv.armors_df = DataFrame(armors_color2num2count)
        return rv

    def __add__(self, other: "ROCODatasetStats") -> "ROCODatasetStats":
        if self.armors_df.empty:
            armors_df = other.armors_df.copy()
        else:
            armors_df = self.armors_df + other.armors_df
        return ROCODatasetStats(
            n_robots=self.n_robots + other.n_robots,
            n_images=self.n_images + other.n_images,
            n_watchers=self.n_watchers + other.n_watchers,
            n_runes=self.n_runes + other.n_runes,
            n_bases=self.n_bases + other.n_bases,
            armors_df=armors_df,
        )


def make_dataset_report(builder: ROCODatasetBuilder) -> ROCODatasetStats:
    stats = ROCODatasetStats.from_dataset(builder.build_lazy())

    with MarkdownFile(builder.main_dir) as mf:
        mf.title(f"Dataset {builder.name}")
        _report_single_stat(mf, stats)

    return stats


def make_datasets_report(datasets: Type[ROCODatasets]):
    name2stats = {builder.name: make_dataset_report(builder) for builder in datasets}
    global_stat = sum(name2stats.values(), ROCODatasetStats())

    with MarkdownFile(datasets.main_dir / "report.md") as mf:
        mf.title("Global report")
        _report_single_stat(mf, global_stat)

        mf.title("Repartition")
        mf.table(add_percentages_to_df(DataFrame({name: stat.numbers for name, stat in name2stats.items()}), axis=1))

        print(mf)


def _report_single_stat(mf: MarkdownFile, stats: ROCODatasetStats):
    mf.paragraph(f"{stats.n_images} images, with:")
    mf.list([f"{number} {name}" for name, number in stats.numbers.items()])
    mf.table(DataFrame(stats.armors_df))


if __name__ == "__main__":
    apply(make_datasets_report, ROCODatasetsZoo)
