from polystar.utils.serialization import pkl_load
from polystar.utils.time import create_time_id
from research.armors.armor_digit.pipeline import ArmorDigitKerasPipeline
from research.armors.armor_digit.training import train_report_and_upload_digit_pipeline
from research.common.constants import PIPELINES_DIR
from research.common.utils.logs import setup_dev_logs

if __name__ == "__main__":
    setup_dev_logs()

    _training_dir = PIPELINES_DIR / "armor-digit" / f"{create_time_id()}_kd_cnn"

    _kd_cnn_pipeline = ArmorDigitKerasPipeline.from_distillation(
        teacher_pipeline=pkl_load(
            PIPELINES_DIR / "armor-digit/20210110_220816_vgg16_full_dset/wrapper (32) - lr 2.1e-04 - drop 0.pkl"
        ),
        conv_blocks=((32, 32), (64, 64)),
        logs_dir=str(_training_dir),
        dropout=0.63,
        lr=0.000776,
        dense_size=1024,
        temperature=41.2,
        verbose=1,
    )

    train_report_and_upload_digit_pipeline(_kd_cnn_pipeline, training_dir=_training_dir)
