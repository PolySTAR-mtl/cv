from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
from xml.dom.minidom import parseString

import xmltodict
from dicttoxml import dicttoxml

from polystar.common.models.image import Image


@dataclass
class PointOfInterest:
    x: int
    y: int

    def to_dict(self) -> Dict[str, int]:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(x=int(d["x"]), y=int(d["y"]))

    @classmethod
    def from_annotation_file(cls, annotation_path: Path) -> List["PointOfInterest"]:
        points = xmltodict.parse(annotation_path.read_text())["annotation"]["point"]
        return [PointOfInterest.from_dict(p) for p in points]


@dataclass
class LabeledImage:
    image: Image
    point_of_interests: List[PointOfInterest] = field(default_factory=list)

    def save(self, directory_path: Path, name: str):
        Image.save(self.image, directory_path / "image" / f"{name}.jpg")
        self._save_annotation(directory_path / "image_annotation" / f"{name}.xml")

    def _save_annotation(self, annotation_path: Path):
        annotation_path.parent.mkdir(exist_ok=True, parents=True)
        xml = parseString(
            dicttoxml(
                {"annotation": {"point": [p.to_dict() for p in self.point_of_interests]}},
                attr_type=False,
                root="annotation",
                item_func=lambda x: x,
            )
            .replace(b"<point><point>", b"<point>")
            .replace(b"</point></point>", b"</point>")
        ).toprettyxml()
        annotation_path.write_text(xml)

    @staticmethod
    def from_directory(directory_path: Path, name: str) -> "LabeledImage":
        return LabeledImage(
            image=Image.from_path(directory_path / "image" / f"{name}.jpg"),
            point_of_interests=PointOfInterest.from_annotation_file(
                directory_path / "image_annotation" / f"{name}.xml"
            ),
        )
