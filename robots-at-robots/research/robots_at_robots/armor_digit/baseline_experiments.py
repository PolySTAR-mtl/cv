import logging

from polystar.common.models.object import ArmorDigit
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.pipeline.classification.random_model import RandomClassifier
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.armor_digit.armor_digit_pipeline_reporter_factory import (
    ArmorDigitPipelineReporterFactory,
)


class ArmorTypePipeline(ClassificationPipeline):
    enum = ArmorDigit


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    reporter = ArmorDigitPipelineReporterFactory.from_roco_datasets(
        train_roco_datasets=[ROCODatasetsZoo.TWITCH.T470151286, ROCODatasetsZoo.TWITCH.T470150052],
        test_roco_datasets=[ROCODatasetsZoo.TWITCH.T470152289],
    )

    random_pipeline = ArmorTypePipeline.from_pipes([RandomClassifier()], name="random")

    reporter.report([random_pipeline], evaluation_short_name="baseline")
