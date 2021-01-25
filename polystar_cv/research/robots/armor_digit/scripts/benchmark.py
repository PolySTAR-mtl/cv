import logging
import warnings
from pathlib import Path

from polystar.common.pipeline.classification.random_model import RandomClassifier
from polystar.common.utils.serialization import pkl_load
from research.common.utils.experiment_dir import prompt_experiment_dir
from research.robots.armor_digit.digit_benchmarker import make_default_digit_benchmarker
from research.robots.armor_digit.pipeline import ArmorDigitKerasPipeline, ArmorDigitPipeline

if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")

    _report_dir: Path = prompt_experiment_dir("armor-digit")

    logging.info(f"Running benchmarking {_report_dir.name}")

    _benchmarker = make_default_digit_benchmarker(report_dir=_report_dir)

    _random_pipeline = ArmorDigitPipeline.from_pipes([RandomClassifier()], name="random")
    _cnn_pipeline = ArmorDigitKerasPipeline.from_custom_cnn(
        input_size=32,
        conv_blocks=((32, 32), (64, 64)),
        logs_dir=str(_report_dir),
        dropout=0.66,
        lr=0.00078,
        dense_size=1024,
        name="cnn",
    )
    # _vgg16_pipeline = ArmorDigitKerasPipeline.from_transfer_learning(
    #     input_size=32, logs_dir=_report_dir, dropout=0, lr=0.00021, dense_size=64, model_factory=VGG16
    # )

    _vgg16_pipeline = pkl_load(PIPELINES_DIR / "armor-digit/20201225_131957_vgg16/VGG16 (32) - lr 2.1e-04 - drop 0.pkl")
    _vgg16_pipeline.name = "vgg16_tl"

    _distiled_vgg16_into_cnn_pipeline = ArmorDigitKerasPipeline.from_distillation(
        teacher_pipeline=_vgg16_pipeline,
        conv_blocks=((32, 32), (64, 64)),
        logs_dir=_report_dir,
        dropout=0.63,
        lr=0.000776,
        dense_size=1024,
        temperature=41.2,
        name="cnn_kd",
    )

    _benchmarker.benchmark(
        pipelines=[_random_pipeline, _distiled_vgg16_into_cnn_pipeline, _cnn_pipeline],
        trained_pipelines=[_vgg16_pipeline],
    )
