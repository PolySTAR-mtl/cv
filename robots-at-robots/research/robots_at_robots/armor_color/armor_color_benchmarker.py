from typing import List

from polystar.common.models.object import ArmorColor
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.robots_at_robots.armor_color.armor_color_dataset import make_armor_color_dataset_generator
from research.robots_at_robots.evaluation.benchmark import make_armor_value_benchmarker


def make_armor_color_benchmarker(
    train_roco_datasets: List[ROCODatasetBuilder], test_roco_datasets: List[ROCODatasetBuilder], experiment_name: str
):
    dataset_generator = make_armor_color_dataset_generator()
    return make_armor_value_benchmarker(
        train_roco_datasets=train_roco_datasets,
        test_roco_datasets=test_roco_datasets,
        evaluation_project="armor-color",
        experiment_name=experiment_name,
        classes=list(ArmorColor),
        dataset_generator=dataset_generator,
    )
