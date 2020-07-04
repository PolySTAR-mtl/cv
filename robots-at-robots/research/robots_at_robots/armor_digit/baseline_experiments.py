import logging

from polystar.common.image_pipeline.classifier_image_pipeline import ClassifierImagePipeline
from polystar.common.image_pipeline.models.random_model import RandomModel
from research.common.dataset.twitch.twitch_roco_datasets import TwitchROCODataset
from research.robots_at_robots.armor_digit.armor_digit_pipeline_reporter_factory import \
    ArmorDigitPipelineReporterFactory

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    reporter = ArmorDigitPipelineReporterFactory.from_roco_datasets(
        train_roco_datasets=[TwitchROCODataset.TWITCH_470151286, TwitchROCODataset.TWITCH_470150052],
        test_roco_datasets=[TwitchROCODataset.TWITCH_470152289],
    )

    random_pipeline = ClassifierImagePipeline(model=RandomModel(), custom_name="random")

    reporter.report([random_pipeline], evaluation_short_name="baseline")
