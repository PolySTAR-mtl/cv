from pathlib import Path
from time import time
from typing import Iterable

import seaborn
from matplotlib.pyplot import show, title
from pandas import DataFrame

from polystar.models.image import FileImage, file_images_to_images
from polystar.models.roco_object import ArmorDigit
from polystar.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.utils.iterable_utils import chunk
from polystar.utils.serialization import pkl_load
from research.armors.armor_digit.armor_digit_dataset import make_armor_digit_dataset_generator
from research.common.constants import PIPELINES_DIR
from research.common.datasets.dataset import Dataset
from research.common.gcloud.gcloud_storage import GCStorages
from research.common.utils.logs import setup_dev_logs


def time_digit_pipeline(pipeline_path: Path):
    seaborn.set()
    GCStorages.DEV.download_file_if_missing(pipeline_path)
    pipeline = pkl_load(pipeline_path)
    test_datasets = make_armor_digit_dataset_generator().default_test_datasets()
    pipeline.predict_proba_and_classes(file_images_to_images(test_datasets[0].examples[:5]))
    df = DataFrame(
        [
            {"dataset": dataset.name, "time (s)": t, "batch_size": batch_size}
            for dataset in test_datasets
            for batch_size in range(1, 16)
            for t in time_batch_inference(pipeline, dataset, batch_size)
        ]
    )
    seaborn.violinplot(x="batch_size", y="time (s)", data=df, cut=0)
    title(f"Time inference\n{pipeline.name}")
    show()


def time_batch_inference(
    pipeline: ClassificationPipeline, dataset: Dataset[FileImage, ArmorDigit], batch_size: int
) -> Iterable[float]:
    for file_images in chunk(dataset.examples, batch_size):
        images = file_images_to_images(file_images)
        t = time()
        pipeline.predict_proba_and_classes(images)
        rv = time() - t
        if rv > 0.1:
            print(rv, *[f"file://{f.path}" for f in file_images])
        yield rv


if __name__ == "__main__":
    setup_dev_logs()
    time_digit_pipeline(
        PIPELINES_DIR
        / "armor-digit/20210117_145856_kd_cnn/distiled - temp 4.1e+01 - cnn - (32) - lr 7.8e-04 - drop 63.pkl"
    )
    time_digit_pipeline(
        PIPELINES_DIR / "armor-digit/20210110_220816_vgg16_full_dset/wrapper (32) - lr 2.1e-04 - drop 0.pkl"
    )
