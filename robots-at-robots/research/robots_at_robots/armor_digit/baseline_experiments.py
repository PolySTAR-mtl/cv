import logging

from polystar.common.image_pipeline.classifier_image_pipeline import ClassifierImagePipeline
from polystar.common.image_pipeline.models.random_model import RandomModel
from research.common.datasets_v3.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.armor_digit.armor_digit_pipeline_reporter_factory import (
    ArmorDigitPipelineReporterFactory,
)

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    reporter = ArmorDigitPipelineReporterFactory.from_roco_datasets(
        train_roco_datasets=[ROCODatasetsZoo.TWITCH.T470151286.builder, ROCODatasetsZoo.TWITCH.T470150052.builder],
        test_roco_datasets=[ROCODatasetsZoo.TWITCH.T470152289.builder],
    )

    random_pipeline = ClassifierImagePipeline(model=RandomModel(), custom_name="random")

    reporter.report([random_pipeline], evaluation_short_name="baseline")
