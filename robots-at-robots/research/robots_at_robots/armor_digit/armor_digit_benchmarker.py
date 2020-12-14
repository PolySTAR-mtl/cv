from typing import List

from polystar.common.models.object import ArmorDigit
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.robots_at_robots.armor_digit.armor_digit_dataset import make_armor_digit_dataset_generator
from research.robots_at_robots.evaluation.benchmark import make_armor_value_benchmarker


def make_armor_digit_benchmarker(
    train_roco_datasets: List[ROCODatasetBuilder],
    validation_roco_datasets: List[ROCODatasetBuilder],
    test_roco_datasets: List[ROCODatasetBuilder],
    experiment_name: str,
):
    dataset_generator = make_armor_digit_dataset_generator()
    return make_armor_value_benchmarker(
        train_roco_datasets=train_roco_datasets,
        validation_roco_datasets=validation_roco_datasets,
        test_roco_datasets=test_roco_datasets,
        evaluation_project="armor-digit",
        experiment_name=experiment_name,
        classes=list(ArmorDigit),
        dataset_generator=dataset_generator,
    )
