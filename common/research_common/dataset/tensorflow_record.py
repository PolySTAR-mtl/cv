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
from research_common.constants import TENSORFLOW_RECORDS_DIR
from research_common.dataset.dataset import Dataset


class TensorflowExampleFactory:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.label_map = get_label_map_dict(str(TENSORFLOW_RECORDS_DIR / "label_map.pbtxt"))

    def from_annotation_path(self, annotation_path: Path) -> tf.train.Example:
        annotation = self._load_annotation(annotation_path)
        return self.from_annotation(annotation, annotation_path.stem)

    def from_annotation(self, annotation: Dict[str, Any], img_name: str) -> tf.train.Example:
        full_path = (self.dataset.images_dir_path / img_name).with_suffix(".jpg")
        encoded_jpg = full_path.read_bytes()
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width = int(annotation["size"]["width"])
        height = int(annotation["size"]["height"])

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        for obj in annotation.get("object", []):
            xmin.append(float(obj["bndbox"]["xmin"]) / width)
            ymin.append(float(obj["bndbox"]["ymin"]) / height)
            xmax.append(float(obj["bndbox"]["xmax"]) / width)
            ymax.append(float(obj["bndbox"]["ymax"]) / height)
            classes_text.append(obj["name"].encode("utf8"))
            classes.append(self.label_map[obj["name"]])

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
    writer = python_io.TFRecordWriter(str(TENSORFLOW_RECORDS_DIR / f"{name}_.record"))
    for dataset in datasets:
        example_factory = TensorflowExampleFactory(dataset)
        for annotation_path in tqdm(dataset.annotation_paths, desc=dataset.dataset_name):
            writer.write(example_factory.from_annotation_path(annotation_path).SerializeToString())
    writer.close()


def create_tf_record_from_dataset(dataset: Dataset):
    create_tf_record_from_datasets([dataset], name=dataset.dataset_name)
