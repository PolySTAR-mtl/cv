import logging
import warnings
from pathlib import Path

from optuna import Trial

from research.armors.armor_digit.digit_benchmarker import make_default_digit_benchmarker
from research.armors.armor_digit.pipeline import ArmorDigitKerasPipeline, ArmorDigitPipeline
from research.armors.evaluation.hyper_tuner import HyperTuner
from research.common.utils.experiment_dir import make_experiment_dir


def cnn_pipeline_factory(report_dir: Path, trial: Trial) -> ArmorDigitPipeline:
    return ArmorDigitKerasPipeline.from_custom_cnn(
        input_size=32,
        conv_blocks=((32, 32), (64, 64)),
        logs_dir=str(report_dir),
        dropout=trial.suggest_uniform("dropout", 0, 0.99),
        lr=trial.suggest_loguniform("lr", 1e-5, 1e-1),
        dense_size=2 ** round(trial.suggest_discrete_uniform("dense_size_log2", 3, 10, 1)),
    )


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")

    logging.info("Hyperparameter tuning for CNN pipeline on digit task")
    HyperTuner(make_default_digit_benchmarker(make_experiment_dir("armor-digit", "cnn_tuning"))).tune(
        cnn_pipeline_factory, n_trials=50
    )
