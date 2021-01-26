import pickle
from pathlib import Path

from research.armors.armor_digit.digit_benchmarker import make_default_digit_benchmarker
from research.armors.armor_digit.pipeline import ArmorDigitPipeline
from research.common.gcloud.gcloud_storage import GCStorages


def train_report_and_upload_digit_pipeline(pipeline: ArmorDigitPipeline, training_dir: Path):
    make_default_digit_benchmarker(training_dir).benchmark([pipeline])

    with GCStorages.DEV.open((training_dir / pipeline.name).with_suffix(".pkl"), "wb") as f:
        pickle.dump(pipeline, f)
