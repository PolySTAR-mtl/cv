from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List
from xml.dom.minidom import parseString

import cv2
import xmltodict
from dicttoxml import dicttoxml

from polystar.models.image import Image, load_image, save_image


@dataclass
class PointOfInterest:
    x: int
    y: int
    label: str

    def to_dict(self) -> Dict[str, int]:
        return self.__dict__

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        return cls(x=int(d["x"]), y=int(d["y"]), label=d["label"])

    @classmethod
    def from_annotation_file(cls, annotation_path: Path) -> List["PointOfInterest"]:
        points = xmltodict.parse(annotation_path.read_text())["annotation"]["point"]
        return [PointOfInterest.from_dict(p) for p in points]


@dataclass
class LabeledImage:
    image: Image
    point_of_interests: List[PointOfInterest] = field(default_factory=list)

    def save(self, directory_path: Path, name: str):
        save_image(self.image, directory_path / "image" / f"{name}.jpg")
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
    def from_directory(
        directory: Path, name: str, conversion: int = cv2.COLOR_BGR2RGB, ext: str = "jpg"
    ) -> "LabeledImage":
        return LabeledImage(
            image=load_image(directory / "image" / f"{name}.{ext}", conversion),
            point_of_interests=PointOfInterest.from_annotation_file(directory / "image_annotation" / f"{name}.xml"),
        )


def load_labeled_images_in_directory(
    directory: Path, conversion: int = cv2.COLOR_BGR2RGB, ext: str = "jpg"
) -> Iterable[LabeledImage]:
    for xml_path in directory.glob("image_annotation/*.xml"):
        yield LabeledImage.from_directory(directory, xml_path.stem, conversion, ext)
