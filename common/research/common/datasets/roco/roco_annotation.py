import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import xmltodict
from polystar.common.models.object import Object, ObjectFactory


@dataclass
class ROCOAnnotation:
    objects: List[Object]

    has_rune: bool

    @staticmethod
    def from_xml_file(xml_file: Path) -> "ROCOAnnotation":
        try:
            return ROCOAnnotation.from_xml_dict(xmltodict.parse(xml_file.read_text())["annotation"])
        except Exception as e:
            logging.exception(f"Error parsing annotation file {xml_file}")
            raise e

    @staticmethod
    def from_xml_dict(xml_dict: Dict) -> "ROCOAnnotation":
        json_objects = xml_dict.get("object", []) or []
        json_objects = json_objects if isinstance(json_objects, list) else [json_objects]
        roco_json_objects = [obj_json for obj_json in json_objects if not obj_json["name"].startswith("rune")]
        objects = [ObjectFactory.from_json(obj_json) for obj_json in roco_json_objects]

        return ROCOAnnotation(objects=objects, has_rune=len(roco_json_objects) != len(json_objects))
