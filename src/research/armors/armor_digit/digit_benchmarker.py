from pathlib import Path

from research.armors.armor_digit.armor_digit_dataset import default_armor_digit_datasets
from research.armors.armor_digit.pipeline import ArmorDigitPipeline
from research.armors.evaluation.benchmarker import Benchmarker


def make_default_digit_benchmarker(report_dir: Path) -> Benchmarker:
    train_datasets, validation_datasets, test_datasets = default_armor_digit_datasets()
    return Benchmarker(
        report_dir=report_dir,
        train_datasets=train_datasets,
        validation_datasets=validation_datasets,
        test_datasets=test_datasets,
        classes=ArmorDigitPipeline.classes,
    )
