from pathlib import Path

from polystar.utils.time import create_time_id
from research.constants import EVALUATION_DIR


def prompt_experiment_dir(project_name: str) -> Path:
    experiment_name: str = input(f"Experiment name for {project_name}: ")
    return make_experiment_dir(project_name, experiment_name)


def make_experiment_dir(project_name: str, experiment_name: str) -> Path:
    experiment_dir = EVALUATION_DIR / project_name / f"{create_time_id()}_{experiment_name}"
    experiment_dir.mkdir(exist_ok=True, parents=True)
    return experiment_dir
