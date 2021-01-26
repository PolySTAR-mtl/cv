import pickle
from os.path import join

from research.armors.armor_digit.armor_digit_dataset import default_armor_digit_datasets
from research.armors.armor_digit.pipeline import ArmorDigitPipeline
from research.armors.evaluation.evaluator import ImageClassificationPipelineEvaluator
from research.armors.evaluation.trainer import ImageClassificationPipelineTrainer
from research.common.gcloud.gcloud_storage import GCStorage


def train_evaluate_digit_pipeline(pipeline: ArmorDigitPipeline, job_dir: str):
    train_datasets, val_datasets, test_datasets = default_armor_digit_datasets()
    trainer = ImageClassificationPipelineTrainer(train_datasets, val_datasets)
    evaluator = ImageClassificationPipelineEvaluator(train_datasets, val_datasets, test_datasets)

    trainer.train_pipeline(pipeline)

    with GCStorage.open_from_str(join(job_dir, pipeline.name, "perfs.pkl"), "wb") as f:
        pickle.dump(evaluator.evaluate_pipeline(pipeline), f)
