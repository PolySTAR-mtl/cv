from typing import Generic, List

from tqdm import tqdm

from polystar.common.models.image import file_images_to_images
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from research.common.datasets.image_dataset import FileImageDataset
from research.common.datasets.lazy_dataset import TargetT
from research.common.datasets.union_dataset import UnionDataset


class ImageClassificationPipelineTrainer(Generic[TargetT]):
    def __init__(self, training_datasets: List[FileImageDataset]):
        train_dataset = UnionDataset(training_datasets)
        self.images = file_images_to_images(train_dataset.examples)
        self.labels = train_dataset.targets

    def train_pipeline(self, pipeline: ClassificationPipeline):
        pipeline.fit(self.images, self.labels)

    def train_pipelines(self, pipelines: List[ClassificationPipeline]):
        tqdm_pipelines = tqdm(pipelines, desc="Training Pipelines")
        for pipeline in tqdm_pipelines:
            tqdm_pipelines.set_postfix({"pipeline": pipeline.name}, True)
            self.train_pipeline(pipeline)
