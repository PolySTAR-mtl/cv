import logging

from polystar.common.image_pipeline.classifier_image_pipeline import ClassifierImagePipeline
from polystar.common.image_pipeline.image_featurizer.mean_rgb_channels_featurizer import MeanChannelsFeaturizer
from polystar.common.image_pipeline.models.random_model import RandomModel
from polystar.common.image_pipeline.models.red_blue_channels_comparison_model import RedBlueComparisonModel
from research.armor_color.armor_color_pipeline_reporter_factory import ArmorColorPipelineReporterFactory
from research_common.dataset.twitch.twitch_roco_datasets import TwitchROCODataset

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    reporter = ArmorColorPipelineReporterFactory.from_roco_datasets(
        train_roco_datasets=[TwitchROCODataset.TWITCH_470151286, TwitchROCODataset.TWITCH_470150052],
        test_roco_datasets=[TwitchROCODataset.TWITCH_470152289],
    )

    red_blue_comparison_pipeline = ClassifierImagePipeline(
        image_featurizer=MeanChannelsFeaturizer(),
        model=RedBlueComparisonModel(red_channel_id=0, blue_channel_id=2),
        custom_name="rb-comparison",
    )
    random_pipeline = ClassifierImagePipeline(model=RandomModel(), custom_name="random")

    reporter.report([random_pipeline, red_blue_comparison_pipeline], evaluation_short_name="baselines")
