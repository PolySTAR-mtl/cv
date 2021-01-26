import logging
import warnings
from pathlib import Path

from optuna import Trial

from polystar.utils.serialization import pkl_load
from research.common.utils.experiment_dir import make_experiment_dir
from research.robots.armor_digit.digit_benchmarker import make_default_digit_benchmarker
from research.robots.armor_digit.pipeline import ArmorDigitKerasPipeline, ArmorDigitPipeline
from research.robots.evaluation.hyper_tuner import HyperTuner


class DistilledPipelineFactory:
    def __init__(self, teacher_name: str):
        self.teacher: ArmorDigitKerasPipeline = pkl_load(PIPELINES_DIR / "armor-digit" / teacher_name)

    def __call__(self, report_dir: Path, trial: Trial) -> ArmorDigitPipeline:
        return ArmorDigitKerasPipeline.from_distillation(
            teacher_pipeline=self.teacher,
            conv_blocks=((32, 32), (64, 64)),
            logs_dir=str(report_dir),
            dropout=trial.suggest_uniform("dropout", 0, 0.99),
            lr=trial.suggest_loguniform("lr", 5e-4, 1e-3),
            dense_size=1024,  # 2 ** round(trial.suggest_discrete_uniform("dense_size_log2", 3, 10, 1)),
            temperature=trial.suggest_loguniform("temperature", 1, 100),
        )


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")

    logging.info("Hyperparameter tuning for VGG16 distilation into CNN pipeline on digit task")
    HyperTuner(make_default_digit_benchmarker(make_experiment_dir("armor-digit", "distillation_tuning"))).tune(
        DistilledPipelineFactory("20201225_131957_vgg16/VGG16 (32) - lr 2.1e-04 - drop 0.pkl"), n_trials=50
    )
