import hashlib
from dataclasses import dataclass
from typing import Iterable

import tensorflow as tf
from tensorflow_core.python.lib.io import python_io
from tqdm import tqdm

from object_detection.utils.dataset_util import (
    float_list_feature,
    bytes_feature,
    int64_feature,
    bytes_list_feature,
    int64_list_feature,
)
from polystar.common.models.image_annotation import ImageAnnotation
from polystar.common.utils.tensorflow import LabelMap
from research_common.constants import TENSORFLOW_RECORDS_DIR
from research_common.dataset.dataset import Dataset


@dataclass
class TensorflowRecordFactory:
    label_map: LabelMap

    def from_datasets(self, datasets: Iterable[Dataset], name: str):
        writer = python_io.TFRecordWriter(str(TENSORFLOW_RECORDS_DIR / f"{name}.record"))
        for dataset in datasets:
            for image_annotation in tqdm(dataset.image_annotations, desc=dataset.dataset_name):
                writer.write(self.example_from_image_annotation(image_annotation).SerializeToString())
        writer.close()

    def from_dataset(self, dataset: Dataset):
        self.from_datasets([dataset], name=dataset.dataset_name)

    def example_from_image_annotation(self, image_annotation: ImageAnnotation) -> tf.train.Example:
        encoded_jpg = image_annotation.image_path.read_bytes()
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width, height = image_annotation.width, image_annotation.height

        xmin, ymin, xmax, ymax, classes, classes_text = [], [], [], [], [], []

        for obj in image_annotation.objects:
            xmin.append(float(obj.x) / width)
            ymin.append(float(obj.y) / height)
            xmax.append(float(obj.x + obj.w) / width)
            ymax.append(float(obj.y + obj.h) / height)
            classes_text.append(obj.type.name.lower().encode("utf8"))
            classes.append(self.label_map.id_of(obj.type.name.lower()))

        return tf.train.Example(
            features=tf.train.Features(
                feature={
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
