from typing import Generic, Iterable, List

from tqdm import tqdm

from polystar.common.models.image import file_images_to_images
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from research.common.datasets.image_dataset import FileImageDataset
from research.common.datasets.lazy_dataset import TargetT
from research.common.datasets.union_dataset import UnionDataset


class ImageClassificationPipelineTrainer(Generic[TargetT]):
    def __init__(self, training_datasets: List[FileImageDataset], validation_datasets: List[FileImageDataset]):
        dataset = UnionDataset(training_datasets + validation_datasets)
        self.validation_size = sum(len(d) for d in validation_datasets)
        self.images = file_images_to_images(dataset.examples)
        self.labels = dataset.targets

    def train_pipeline(self, pipeline: ClassificationPipeline):
        pipeline.fit(self.images, self.labels, validation_size=self.validation_size)

    def train_pipelines(self, pipelines: Iterable[ClassificationPipeline]):
        tqdm_pipelines = tqdm(pipelines, desc="Training Pipelines")
        for pipeline in tqdm_pipelines:
            tqdm_pipelines.set_postfix({"pipeline": pipeline.name}, True)
            self.train_pipeline(pipeline)
