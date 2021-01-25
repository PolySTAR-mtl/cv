from tensorflow.python.keras.applications.vgg16 import VGG16

from polystar.common.utils.time import create_time_id
from research.common.constants import PIPELINES_DIR
from research.common.utils.logs import setup_dev_logs
from research.robots.armor_digit.pipeline import ArmorDigitKerasPipeline
from research.robots.armor_digit.training import train_report_and_upload_digit_pipeline

if __name__ == "__main__":
    setup_dev_logs()

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

    train_report_and_upload_digit_pipeline(_vgg16_pipeline, _training_dir)
