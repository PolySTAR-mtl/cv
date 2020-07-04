import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from xml.dom.minidom import parseString

import xmltodict
from dicttoxml import dicttoxml
from polystar.common.models.image import Image
from polystar.common.models.object import Object, ObjectFactory


@dataclass
class ImageAnnotation:

    image_path: Path
    xml_path: Path

    width: int
    height: int

    objects: List[Object]

    has_rune: bool

    _image: Image = field(repr=False, default=None)

    @property
    def image(self) -> Image:
        if self._image is None:
            self._image = Image.from_path(self.image_path)
        return self._image

    @staticmethod
    def from_xml_file(xml_file: Path) -> "ImageAnnotation":
        try:
            annotation = xmltodict.parse(xml_file.read_text())["annotation"]

            json_objects = annotation.get("object", []) or []
            json_objects = json_objects if isinstance(json_objects, list) else [json_objects]
            roco_json_objects = [obj_json for obj_json in json_objects if not obj_json["name"].startswith("rune")]
            objects = [ObjectFactory.from_json(obj_json) for obj_json in roco_json_objects]

            return ImageAnnotation(
                width=int(annotation["size"]["width"]),
                height=int(annotation["size"]["height"]),
                objects=objects,
                image_path=xml_file.parent.parent / "image" / f"{xml_file.stem}.jpg",
                xml_path=xml_file,
                has_rune=len(roco_json_objects) != len(json_objects),
            )
        except Exception as e:
            logging.error(f"Error parsing annotation file {xml_file}")
            logging.exception(e)
            raise e

    def to_xml(self) -> str:
        return parseString(
            dicttoxml(
                {
                    "annotation": {
                        "size": {"width": self.width, "height": self.height},
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

    def save_to_dir(self, directory: Path, image_name: str):
        self.image_path = (directory / "image" / image_name).with_suffix(".jpg")
        self.xml_path = (directory / "image_annotation" / image_name).with_suffix(".xml")

        self.image_path.parent.mkdir(exist_ok=True, parents=True)
        self.xml_path.parent.mkdir(exist_ok=True, parents=True)

        Image.save(self.image, self.image_path)
        self.xml_path.write_text(self.to_xml())
