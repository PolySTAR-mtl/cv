import logging
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from polystar.pipeline.classification.random_model import RandomClassifier
from polystar.pipeline.classification.red_blue_comparison_classifier import RedBlueComparisonClassifier
from polystar.pipeline.featurizers.histogram_2d import Histogram2D
from polystar.pipeline.featurizers.histogram_blocs_2d import HistogramBlocs2D
from polystar.pipeline.featurizers.mean_channels import MeanChannels
from polystar.pipeline.preprocessors.rgb_to_hsv import RGB2HSV
from research.armors.armor_color.benchmarker import make_armor_color_benchmarker
from research.armors.armor_color.pipeline import ArmorColorPipeline
from research.common.utils.experiment_dir import prompt_experiment_dir

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.info("Benchmarking")

    _report_dir: Path = prompt_experiment_dir("armor-color")
    _benchmarker = make_armor_color_benchmarker(_report_dir, include_dji=False)

    _pipelines = [
        ArmorColorPipeline.from_pipes([MeanChannels(), RedBlueComparisonClassifier()], name="rb-comparison"),
        ArmorColorPipeline.from_pipes([RandomClassifier()], name="random"),
        ArmorColorPipeline.from_pipes(
            [RGB2HSV(), Histogram2D(), LogisticRegression(max_iter=200)], name="hsv-hist-lr",
        ),
        ArmorColorPipeline.from_pipes(
            [RGB2HSV(), HistogramBlocs2D(rows=1, cols=3), LogisticRegression(max_iter=200)], name="hsv-hist-blocs-lr",
        ),
        ArmorColorPipeline.from_pipes([Histogram2D(), LogisticRegression(max_iter=200)], name="rgb-hist-lr"),
        # ArmorColorKerasPipeline.from_custom_cnn(
        #     logs_dir=str(_report_dir),
        #     input_size=16,
        #     conv_blocks=((32, 32), (64, 64)),
        #     dropout=0.5,
        #     dense_size=64,
        #     lr=7.2e-4,
        #     name="cnn",
        #     batch_size=128,
        #     steps_per_epoch="auto",
        # ),
    ]

    _benchmarker.benchmark(_pipelines)
