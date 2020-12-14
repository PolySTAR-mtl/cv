import logging
from dataclasses import dataclass

from nptyping import Array
from sklearn.linear_model import LogisticRegression

from polystar.common.image_pipeline.featurizers.histogram_2d import Histogram2D
from polystar.common.image_pipeline.preprocessors.rgb_to_hsv import RGB2HSV
from polystar.common.models.image import Image
from polystar.common.models.object import ArmorColor
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.pipeline.classification.random_model import RandomClassifier
from polystar.common.pipeline.classification.rule_based_classifier import RuleBasedClassifierABC
from polystar.common.pipeline.pipe_abc import PipeABC
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.armor_color.armor_color_benchmarker import make_armor_color_benchmarker


class ArmorColorPipeline(ClassificationPipeline):
    enum = ArmorColor


@dataclass
class MeanChannels(PipeABC):
    def transform_single(self, image: Image) -> Array[float, float, float]:
        return image.mean(axis=(0, 1))


class RedBlueComparisonClassifier(RuleBasedClassifierABC):
    """A very simple model that compares the blue and red values obtained by the MeanChannels"""

    def predict_single(self, features: Array[float, float, float]) -> ArmorColor:
        return ArmorColor.Red if features[0] >= features[2] else ArmorColor.Blue


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    _benchmarker = make_armor_color_benchmarker(
        train_roco_datasets=[
            ROCODatasetsZoo.TWITCH.T470150052,
            ROCODatasetsZoo.TWITCH.T470152289,
            ROCODatasetsZoo.TWITCH.T470149568,
            ROCODatasetsZoo.TWITCH.T470151286,
        ],
        validation_roco_datasets=[],
        test_roco_datasets=[
            ROCODatasetsZoo.TWITCH.T470152838,
            ROCODatasetsZoo.TWITCH.T470153081,
            ROCODatasetsZoo.TWITCH.T470158483,
            ROCODatasetsZoo.TWITCH.T470152730,
        ],
        experiment_name="test",
    )

    red_blue_comparison_pipeline = ArmorColorPipeline.from_pipes(
        [MeanChannels(), RedBlueComparisonClassifier()], name="rb-comparison",
    )
    random_pipeline = ArmorColorPipeline.from_pipes([RandomClassifier()], name="random")
    hsv_hist_lr_pipeline = ArmorColorPipeline.from_pipes(
        [RGB2HSV(), Histogram2D(), LogisticRegression()], name="hsv-hist-lr",
    )

    _benchmarker.benchmark([random_pipeline, red_blue_comparison_pipeline, hsv_hist_lr_pipeline])
