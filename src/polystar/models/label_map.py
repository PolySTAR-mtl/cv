import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from polystar.constants import LABEL_MAP_PATH


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
        matches: List[re.Match] = re.findall(
            r"item *?{" r".*?" r"id: (?P<id>\d)" r".*?" r"name: '(?P<name>\w*)'" r".*?" r"\}",
            file_path.read_text(),
            re.S,
        )
        return LabelMap.from_dict({int(i): n for i, n in matches})

    @staticmethod
    def from_dict(d: Dict[int, Any]) -> "LabelMap":
        name2id = {n: int(i) for i, n in d.items()}
        id2name = d
        return LabelMap(id2name=id2name, name2id=name2id)


label_map = LabelMap.from_file(LABEL_MAP_PATH)
