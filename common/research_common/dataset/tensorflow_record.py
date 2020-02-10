import hashlib
from pathlib import Path
from typing import Dict, Any, Iterable

import tensorflow as tf
from lxml import etree
from tensorflow_core.python.lib.io import python_io
from tqdm import tqdm

from object_detection.utils.dataset_util import (
    float_list_feature,
    bytes_feature,
    int64_feature,
    bytes_list_feature,
    int64_list_feature,
    recursive_parse_xml_to_dict,
)
from object_detection.utils.label_map_util import get_label_map_dict
from polystar.common.models.image_annotation import ImageAnnotation
from research_common.constants import TENSORFLOW_RECORDS_DIR
from research_common.dataset.dataset import Dataset


class TensorflowExampleFactory:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.label_map = get_label_map_dict(str(TENSORFLOW_RECORDS_DIR / "label_map.pbtxt"))

    def from_image_annotation(self, image_annotation: ImageAnnotation) -> tf.train.Example:
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
            classes.append(self.label_map[obj.type.name.lower()])

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

    @staticmethod
    def _load_annotation(annotation_path: Path) -> Dict[str, Any]:
        xml = etree.fromstring(annotation_path.read_text())
        return recursive_parse_xml_to_dict(xml)["annotation"]


def create_tf_record_from_datasets(datasets: Iterable[Dataset], name: str):
    writer = python_io.TFRecordWriter(str(TENSORFLOW_RECORDS_DIR / f"{name}.record"))
    for dataset in datasets:
        example_factory = TensorflowExampleFactory(dataset)
        for image_annotation in tqdm(dataset.image_annotations, desc=dataset.dataset_name):
            writer.write(example_factory.from_image_annotation(image_annotation).SerializeToString())
    writer.close()


def create_tf_record_from_dataset(dataset: Dataset):
    create_tf_record_from_datasets([dataset], name=dataset.dataset_name)
