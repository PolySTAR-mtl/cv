import logging
import warnings

from tensorflow.python.keras.applications.vgg16 import VGG16

from polystar.common.constants import PROJECT_DIR
from polystar.common.utils.serialization import pkl_dump
from polystar.common.utils.time import create_time_id
from research.robots.armor_digit.digit_benchmarker import make_default_digit_benchmarker
from research.robots.armor_digit.pipeline import ArmorDigitKerasPipeline

PIPELINES_DIR = PROJECT_DIR / "pipelines"

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")
    logging.info("Training vgg16")

    _training_dir = PIPELINES_DIR / "armor-digit" / f"{create_time_id()}_vgg16_full_dset"

    _vgg16_pipeline = ArmorDigitKerasPipeline.from_transfer_learning(
        input_size=32,
        logs_dir=str(_training_dir),
        dropout=0,
        lr=0.00021,
        dense_size=64,
        model_factory=VGG16,
        verbose=1,
    )

    logging.info(f"Run `tensorboard --logdir={_training_dir}` for realtime logs")

    _benchmarker = make_default_digit_benchmarker(_training_dir)
    _benchmarker.benchmark([_vgg16_pipeline])

    pkl_dump(_vgg16_pipeline, _training_dir / _vgg16_pipeline.name)
