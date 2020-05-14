import hashlib
from dataclasses import dataclass
from shutil import move
from typing import Iterable

import tensorflow as tf
from tensorflow_core.python.lib.io import python_io
from tqdm import tqdm

from polystar.common.models.image_annotation import ImageAnnotation
from polystar.common.models.label_map import LabelMap
from research_common.constants import TENSORFLOW_RECORDS_DIR
from research_common.dataset.directory_roco_dataset import DirectoryROCODataset
from research_common.dataset.roco_dataset import ROCODataset


@dataclass
class TensorflowRecordFactory:
    label_map: LabelMap

    def from_datasets(self, datasets: Iterable[DirectoryROCODataset], name: str):
        writer = python_io.TFRecordWriter(str(TENSORFLOW_RECORDS_DIR / f"{name}.record"))
        c = 0
        for dataset in datasets:
            for image_annotation in tqdm(dataset.image_annotations, desc=dataset.dataset_name):
                writer.write(self.example_from_image_annotation(image_annotation).SerializeToString())
                c += 1
        writer.close()
        move(str(TENSORFLOW_RECORDS_DIR / f"{name}.record"), str(TENSORFLOW_RECORDS_DIR / f"{name}_{c}_imgs.record"))

    def from_dataset(self, dataset: ROCODataset):
        self.from_datasets([dataset], name=dataset.dataset_name)

    def example_from_image_annotation(self, image_annotation: ImageAnnotation) -> tf.train.Example:
        image_name = image_annotation.image_path.name
        encoded_jpg = image_annotation.image_path.read_bytes()
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width, height = image_annotation.width, image_annotation.height

        xmin, ymin, xmax, ymax, classes, classes_text = [], [], [], [], [], []

        for obj in image_annotation.objects:
            xmin.append(float(obj.box.x1) / width)
            ymin.append(float(obj.box.y1) / height)
            xmax.append(float(obj.box.x2) / width)
            ymax.append(float(obj.box.y2) / height)
            classes_text.append(obj.type.name.lower().encode("utf8"))
            classes.append(self.label_map.id_of(obj.type.name.lower()))

        return tf.train.Example(
            features=tf.train.Features(
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


# Functions copied from https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
