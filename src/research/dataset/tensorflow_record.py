import hashlib
from pathlib import Path
from shutil import move
from typing import List

from tensorflow.core.example.example_pb2 import Example
from tensorflow.core.example.feature_pb2 import BytesList, Feature, Features, FloatList, Int64List
from tensorflow_core.python.lib.io.tf_record import TFRecordWriter

from polystar.filters.pass_through_filter import PassThroughFilter
from polystar.models.label_map import label_map
from polystar.target_pipeline.objects_filters.objects_filter_abc import ObjectsFilterABC
from polystar.utils.tqdm import smart_tqdm
from research.common.constants import TENSORFLOW_RECORDS_DIR
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder


class TensorflowRecordFactory:
    def __init__(self, objects_filter: ObjectsFilterABC = PassThroughFilter()):
        self.objects_filter = objects_filter

    def from_dataset(self, dataset: ROCODatasetBuilder, prefix: str = ""):
        self.from_datasets([dataset], prefix)

    def from_datasets(self, datasets: List[ROCODatasetBuilder], prefix: str = ""):
        record_name = prefix + "_".join(d.name for d in datasets)
        writer = TFRecordWriter(str(TENSORFLOW_RECORDS_DIR / f"{record_name}.record"))
        c = 0
        for dataset in smart_tqdm(datasets, desc=record_name, unit="dataset"):
            for image_path, annotation, _ in smart_tqdm(dataset, desc=dataset.name, unit="img", leave=False):
                writer.write(self._example_from_image_annotation(image_path, annotation).SerializeToString())
                c += 1
        writer.close()
        move(
            str(TENSORFLOW_RECORDS_DIR / f"{record_name}.record"),
            str(TENSORFLOW_RECORDS_DIR / f"{record_name}_{c}_imgs.record"),
        )

    def _example_from_image_annotation(self, image_path: Path, annotation: ROCOAnnotation) -> Example:
        image_name = image_path.name
        encoded_jpg = image_path.read_bytes()
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width, height = annotation.w, annotation.h

        xmin, ymin, xmax, ymax, classes, classes_text = [], [], [], [], [], []

        for obj in self.objects_filter.filter(annotation.objects):
            xmin.append(float(obj.box.x1) / width)
            ymin.append(float(obj.box.y1) / height)
            xmax.append(float(obj.box.x2) / width)
            ymax.append(float(obj.box.y2) / height)
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
