from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from xml.dom.minidom import parseString

import xmltodict
from dicttoxml import dicttoxml

from polystar.common.models.object import Object
from polystar.common.view.object_json_factory import ObjectJsonFactory
from polystar.common.models.image import Image


@dataclass
class ImageAnnotation:

    image_path: Path

    width: int
    height: int

    objects: List[Object]

    _image: Image = field(init=False, repr=False, default=None)

    @property
    def image(self) -> Image:
        if self._image is None:
            self._image = Image.from_path(self.image_path)
        return self._image

    @staticmethod
    def from_xml_file(xml_file: Path) -> ImageAnnotation:
        annotation = xmltodict.parse(xml_file.read_text())["annotation"]

        objects = [ObjectJsonFactory.from_json(obj_json) for obj_json in annotation["object"]]

        return ImageAnnotation(
            width=annotation["size"]["width"],
            height=annotation["size"]["height"],
            objects=objects,
            image_path=xml_file.parent.parent / "image" / f"{xml_file.stem}.jpg",
        )

    def to_xml(self) -> str:
        return parseString(
            dicttoxml(
                {
                    "annotation": {
                        "size": {"width": self.width, "height": self.height},
                        "object": [ObjectJsonFactory.to_json(obj) for obj in self.objects],
                    }
                },
                attr_type=False,
                root="annotation",
                item_func=lambda x: x,
            )
            .replace(b"<object><object>", b"<object>")
            .replace(b"</object></object>", b"</object>")
        ).toprettyxml()
