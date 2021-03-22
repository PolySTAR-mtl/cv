import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from xml.dom.minidom import parseString

import xmltodict
from dicttoxml import dicttoxml

from polystar.models.roco_object import ROCOObject, ROCOObjectFactory
from polystar.utils.path import move_file


@dataclass
class ROCOAnnotation:
    objects: List[ROCOObject]

    has_rune: bool

    w: int
    h: int

    @staticmethod
    def from_xml_file(xml_file: Path) -> "ROCOAnnotation":
        try:
            return ROCOAnnotation.from_xml_dict(xmltodict.parse(xml_file.read_text())["annotation"])
        except Exception as e:
            logging.exception(f"Error parsing annotation file file://{xml_file}")
            raise e

    @staticmethod
    def from_xml_dict(xml_dict: Dict) -> "ROCOAnnotation":
        image_h = int(xml_dict["size"]["height"])
        image_w = int(xml_dict["size"]["width"])

        json_objects = xml_dict.get("object", []) or []
        json_objects = json_objects if isinstance(json_objects, list) else [json_objects]
        roco_json_objects = [obj_json for obj_json in json_objects if not obj_json["name"].startswith("rune")]
        objects = [
            ROCOObjectFactory(image_w=image_w, image_h=image_h).from_json(obj_json) for obj_json in roco_json_objects
        ]

        return ROCOAnnotation(
            objects=objects, has_rune=len(roco_json_objects) != len(json_objects), w=image_w, h=image_h,
        )

    def save_in_directory(self, directory: Path, name: str):
        directory.mkdir(exist_ok=True, parents=True)
        self.save_in_file((directory / name).with_suffix(".xml"))

    def save_in_file(self, file: Path):
        file.write_text(self.to_xml())

    def to_xml(self) -> str:
        return parseString(
            dicttoxml(
                {
                    "annotation": {
                        "size": {"width": self.w, "height": self.h},
                        "object": [ROCOObjectFactory.to_json(obj) for obj in self.objects],
                    }
                },
                attr_type=False,
                root="annotation",
                item_func=lambda x: x,
            )
            .replace(b"<object><object>", b"<object>")
            .replace(b"</object></object>", b"</object>")
        ).toprettyxml()


def move_image_and_annotation_from_directory(
    source_dataset_directory: Path, destination_dataset_directory: Path, name: str
):
    move_file((source_dataset_directory / "image" / name).with_suffix(".jpg"), destination_dataset_directory / "image")
    move_file(
        (source_dataset_directory / "image_annotation" / name).with_suffix(".xml"),
        destination_dataset_directory / "image_annotation",
    )
