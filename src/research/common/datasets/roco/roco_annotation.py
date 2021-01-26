import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from xml.dom.minidom import parseString

import xmltodict
from dicttoxml import dicttoxml

from polystar.models.roco_object import Armor, ObjectFactory, ROCOObject


@dataclass
class ROCOAnnotation:
    objects: List[ROCOObject]

    has_rune: bool

    w: int
    h: int

    @property
    def armors(self) -> List[Armor]:
        return [obj for obj in self.objects if isinstance(obj, Armor)]

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

        return ROCOAnnotation(
            objects=objects,
            has_rune=len(roco_json_objects) != len(json_objects),
            w=int(xml_dict["size"]["width"]),
            h=int(xml_dict["size"]["height"]),
        )

    def to_xml(self) -> str:
        return parseString(
            dicttoxml(
                {
                    "annotation": {
                        "size": {"width": self.w, "height": self.h},
                        "object": [ObjectFactory.to_json(obj) for obj in self.objects],
                    }
                },
                attr_type=False,
                root="annotation",
                item_func=lambda x: x,
            )
            .replace(b"<object><object>", b"<object>")
            .replace(b"</object></object>", b"</object>")
        ).toprettyxml()
