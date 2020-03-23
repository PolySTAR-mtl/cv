import logging

from polystar.common.image_pipeline.image_pipeline import ImagePipeline
from polystar.common.image_pipeline.models.random_model import RandomModel
from research.armor_digit.armor_digit_pipeline_reporter_factory import ArmorDigitPipelineReporterFactory
from research_common.dataset.twitch.twitch_roco_datasets import TwitchROCODataset
from research_common.dataset.union_dataset import UnionDataset

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    reporter = ArmorDigitPipelineReporterFactory.from_roco_datasets(
        train_roco_dataset=UnionDataset(TwitchROCODataset.TWITCH_470151286, TwitchROCODataset.TWITCH_470150052),
        test_roco_dataset=TwitchROCODataset.TWITCH_470152289,
    )

    random_pipeline = ImagePipeline(model=RandomModel(), custom_name="random")

    reporter.report([random_pipeline], evaluation_short_name="baseline")
