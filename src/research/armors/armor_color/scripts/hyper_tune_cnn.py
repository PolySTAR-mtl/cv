import logging
import warnings
from pathlib import Path

from optuna import Trial

from research.armors.armor_color.benchmarker import make_armor_color_benchmarker
from research.armors.armor_color.pipeline import ArmorColorKerasPipeline
from research.armors.evaluation.hyper_tuner import HyperTuner
from research.common.utils.experiment_dir import make_experiment_dir


def cnn_pipeline_factory(report_dir: Path, trial: Trial) -> ArmorColorKerasPipeline:
    return ArmorColorKerasPipeline.from_custom_cnn(
        input_size=32,
        conv_blocks=((32, 32), (64, 64)),
        logs_dir=str(report_dir),
        dropout=trial.suggest_uniform("dropout", 0, 0.99),
        lr=trial.suggest_loguniform("lr", 1e-5, 1e-1),
        dense_size=2 ** round(trial.suggest_discrete_uniform("dense_size_log2", 3, 10, 1)),
        batch_size=64,
        steps_per_epoch="auto",
        verbose=0,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")

    logging.info("Hyperparameter tuning for CNN pipeline on color task")
    HyperTuner(make_armor_color_benchmarker(make_experiment_dir("armor-color", "cnn_tuning"), include_dji=False)).tune(
        cnn_pipeline_factory, n_trials=50
    )
