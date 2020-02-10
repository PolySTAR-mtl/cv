from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import tensorflow as tf

from object_detection.utils import ops as utils_ops, label_map_util


@dataclass
class LabelMap:
    id2name: Dict[int, str]
    name2id: Dict[str, int]

    def name_of(self, id: int) -> str:
        return self.id2name[id]

    def id_of(self, name: str) -> int:
        return self.name2id[name]

    @staticmethod
    def from_file(file_path: Path) -> "LabelMap":
        return LabelMap.from_dict(
            label_map_util.create_category_index_from_labelmap(str(file_path), use_display_name=True)
        )

    @staticmethod
    def from_dict(d: Dict[str, Dict[str, Any]]) -> "LabelMap":
        name2id = {name_id["name"]: name_id["id"] for name_id in d.values()}
        id2name = {name_id["id"]: name_id["name"] for name_id in d.values()}
        return LabelMap(id2name=id2name, name2id=name2id)


def patch_tf_v2():
    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1

    # Patch the location of gfile
    tf.gfile = tf.io.gfile
