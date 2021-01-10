import pickle
from os.path import join

from research.common.gcloud.gcloud_storage import GCStorage
from research.robots.armor_digit.armor_digit_dataset import default_armor_digit_datasets
from research.robots.armor_digit.pipeline import ArmorDigitPipeline
from research.robots.evaluation.evaluator import ImageClassificationPipelineEvaluator
from research.robots.evaluation.trainer import ImageClassificationPipelineTrainer


def train_evaluate_digit_pipeline(pipeline: ArmorDigitPipeline, job_dir: str):
    train_datasets, val_datasets, test_datasets = default_armor_digit_datasets()
    trainer = ImageClassificationPipelineTrainer(train_datasets, val_datasets)
    evaluator = ImageClassificationPipelineEvaluator(train_datasets, val_datasets, test_datasets)

    trainer.train_pipeline(pipeline)

    with GCStorage.open_from_str(join(job_dir, pipeline.name, "perfs.pkl"), "wb") as f:
        pickle.dump(evaluator.evaluate_pipeline(pipeline), f)
