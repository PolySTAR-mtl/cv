from pathlib import Path

from research.robots.armor_color.datasets import make_armor_color_datasets
from research.robots.armor_color.pipeline import ArmorColorPipeline
from research.robots.evaluation.benchmarker import Benchmarker


def make_armor_color_benchmarker(report_dir: Path, include_dji: bool = True) -> Benchmarker:
    train_datasets, validation_datasets, test_datasets = make_armor_color_datasets()
    return Benchmarker(
        report_dir=report_dir,
        classes=ArmorColorPipeline.classes,
        train_datasets=train_datasets,
        validation_datasets=validation_datasets,
        test_datasets=test_datasets,
    )
