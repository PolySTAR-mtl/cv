import logging
import pickle
from pathlib import Path
from typing import List

from polystar.common.models.object import ArmorDigit
from polystar.common.utils.iterable_utils import flatten
from research.common.constants import EVALUATION_DIR
from research.common.gcloud.gcloud_storage import GCStorages
from research.robots.evaluation.metrics.f1 import F1Metric
from research.robots.evaluation.performance import ClassificationPerformances
from research.robots.evaluation.reporter import ImagePipelineEvaluationReporter


def load_performances(performances_paths: List[Path]) -> ClassificationPerformances:
    return ClassificationPerformances(flatten(pickle.loads(perf_path.read_bytes()) for perf_path in performances_paths))


def gather_performances(task_name: str, job_id: str):
    logging.info(f"gathering performances for {job_id} on task {task_name}")
    experiment_dir = EVALUATION_DIR / task_name / job_id
    performances_paths = download_performances(experiment_dir)
    performances = load_performances(performances_paths)
    ImagePipelineEvaluationReporter(
        report_dir=EVALUATION_DIR / task_name / job_id, classes=list(ArmorDigit), other_metrics=[F1Metric()]
    ).report(performances)


def download_performances(experiment_dir: Path) -> List[Path]:
    performances_paths = list(GCStorages.DEV.glob(experiment_dir, extension=".pkl"))
    logging.info(f"Found {len(performances_paths)} performances")
    for performance_path in performances_paths:
        GCStorages.DEV.download_file_if_missing(performance_path)
    return performances_paths


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")

    gather_performances("armor-digit", "cnn_20201220_224525")
    gather_performances("armor-digit", "vgg16_20201220_224417")
