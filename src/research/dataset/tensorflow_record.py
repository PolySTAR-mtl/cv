import hashlib
from pathlib import Path
from threading import Thread
from typing import Iterable, List

from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import BytesList, Feature, Features, FloatList, Int64List
from tensorflow_core.python.lib.io.tf_record import TFRecordWriter

from polystar.filters.pass_through_filter import PassThroughFilter
from polystar.models.label_map import label_map
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC
from polystar.utils.iterable_utils import chunk
from polystar.utils.path import make_path
from research.common.constants import TENSORFLOW_RECORDS_DIR
from research.common.datasets.dataset import Dataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.common.datasets.shuffle_dataset import ShuffleDataset
from research.common.datasets.union_dataset import UnionDataset


class TensorflowRecordFactory:
    def __init__(self, objects_filter: ObjectsFilterABC = PassThroughFilter(), n_images_per_file: int = 200):
        self.n_images_per_file = n_images_per_file
        self.objects_filter = objects_filter

    def from_builders(self, builders: List[ROCODatasetBuilder], prefix: str = ""):
        dataset = UnionDataset(d.build() for d in builders)
        dataset.name = f"{prefix}_{dataset.name}_{len(dataset)}_imgs"

        self.from_dataset(dataset)

    def from_dataset(self, dataset: Dataset[Path, ROCOAnnotation]):
        records_dir = make_path(TENSORFLOW_RECORDS_DIR / dataset.name)
        chunks = list(chunk(ShuffleDataset(dataset), self.n_images_per_file))

        for chunk_number, dataset_chunk in enumerate(chunks):
            TFRecordFactoryThread(
                records_dir / f"images_{chunk_number:05}_of_{len(chunks):05}.record", dataset_chunk
            ).start()


class TFRecordFactoryThread(Thread):
    def __init__(self, record_file: Path, dataset_chunk: Iterable):
        super().__init__()
        self.dataset_chunk = dataset_chunk
        self.record_file = record_file

    def run(self) -> None:
        with TFRecordWriter(str(self.record_file)) as writer:
            for image_path, annotation, _ in self.dataset_chunk:
                writer.write(_example_from_image_annotation(image_path, annotation).SerializeToString())


def _example_from_image_annotation(image_path: Path, annotation: ROCOAnnotation) -> Example:
    image_name = image_path.name
    encoded_jpg = image_path.read_bytes()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width, height = annotation.w, annotation.h

    xmin, ymin, xmax, ymax, classes, classes_text = [], [], [], [], [], []

    for obj in annotation.objects:
        x1 = max(0.0, obj.box.x1 / width)
        y1 = max(0.0, obj.box.y1 / height)
        x2 = min(1.0, obj.box.x2 / width)
        y2 = min(1.0, obj.box.y2 / height)
        if x1 >= x2 or y1 >= y2:
            continue
        xmin.append(x1)
        ymin.append(y1)
        xmax.append(x2)
        ymax.append(y2)
        classes_text.append(obj.type.name.lower().encode("utf8"))
        classes.append(label_map.id_of(obj.type.name.lower()))

    return Example(
        features=Features(
            feature={
                "image/filename": bytes_feature(image_name.encode("utf8")),
                "image/source_id": bytes_feature(image_name.encode("utf8")),
                "image/height": int64_feature(height),
                "image/width": int64_feature(width),
                "image/key/sha256": bytes_feature(key.encode("utf8")),
                "image/encoded": bytes_feature(encoded_jpg),
                "image/format": bytes_feature("jpeg".encode("utf8")),
                "image/object/bbox/xmin": float_list_feature(xmin),
                "image/object/bbox/xmax": float_list_feature(xmax),
                "image/object/bbox/ymin": float_list_feature(ymin),
                "image/object/bbox/ymax": float_list_feature(ymax),
                "image/object/class/text": bytes_list_feature(classes_text),
                "image/object/class/label": int64_list_feature(classes),
            }
        )
    )


# Functions inspired from
#  https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
def int64_feature(value: int) -> Feature:
    return int64_list_feature([value])


def int64_list_feature(value: List[int]) -> Feature:
    return Feature(int64_list=Int64List(value=value))


def bytes_feature(value: bytes) -> Feature:
    return bytes_list_feature([value])


def bytes_list_feature(value: List[bytes]) -> Feature:
    return Feature(bytes_list=BytesList(value=value))


def float_list_feature(value: List[float]) -> Feature:
    return Feature(float_list=FloatList(value=value))
