import logging
import re
from pathlib import Path
from typing.re import Match


class AnnotationFileCorrector:
    FINAL_ARMOR_NAME_PATTERN = re.compile(
        r"<name>(armor|amor|amror)-(?P<color>\w{2,4})-(?P<num>\d)</name>", re.IGNORECASE
    )
    ABV_ARMOR_NAME_PATTERN = re.compile(r"<name>a{1,2}(?P<color>\w)(?P<num>\d)</name>", re.IGNORECASE)
    ABV_RUNES_PATTERN = re.compile(r"<name>r(?P<color>\w)</name>", re.IGNORECASE)
    ABV_BASE_PATTERN = re.compile(r"<name>b</name>", re.IGNORECASE)
    ABV_WATCHER_PATTERN = re.compile(r"<name>w</name>", re.IGNORECASE)
    ABV_CAR_PATTERN = re.compile(r"<name>(c|x|o|robot)</name>", re.IGNORECASE)

    COLORS_MAP = {
        "r": "red",
        "red": "red",
        "b": "blue",
        "bleu": "blue",
        "blue": "blue",
        "g": "grey",
        "grey": "grey",
        "gray": "grey",
    }

    def __init__(self, save_before: bool):
        self.save_before = save_before

    def correct_annotations_in_directory(self, directory: Path):
        for annotation_path in directory.glob("*.xml"):
            self.correct_annotation_file(annotation_path)

    def correct_annotation_file(self, annotation_file: Path):
        try:
            text = self._save_previous_content(annotation_file)
            text = self._correct_annotation_text(text)
            annotation_file.write_text(text)
        except Exception as e:
            logging.exception(f"Error processing annotation file://{annotation_file}")
            raise e

    def _correct_annotation_text(self, text: str) -> str:
        text = self._correct_mistakes(text)
        text = self._correct_abbreviations(text)
        text = self._correct_armor_format(text)
        return text

    def _save_previous_content(self, annotation_file: Path) -> str:
        previous_content = annotation_file.read_text()
        if self.save_before:
            save_dir = annotation_file.parent / "save"
            save_dir.mkdir(exist_ok=True)
            (save_dir / annotation_file.name).write_text(previous_content)
        return previous_content

    def _correct_abbreviations(self, text: str) -> str:
        text = self._correct_armor_abbreviation(text)
        text = self._correct_base_abbreviation(text)
        text = self._correct_watcher_abbreviation(text)
        text = self._correct_car_abbreviation(text)
        text = self._correct_runes_abbreviation(text)
        return text

    def _correct_armor_format(self, text: str) -> str:
        return self.FINAL_ARMOR_NAME_PATTERN.sub(self._armor_pattern_mapping, text)

    def _armor_pattern_mapping(self, match: Match):
        color = self.COLORS_MAP[match.group("color")]
        num = match.group("num")
        if num == "p":
            num = 4
        return f"<name>armor</name> <armor_class>{num}</armor_class> <armor_color>{color}</armor_color>"

    def _correct_armor_abbreviation(self, text: str) -> str:
        return self.ABV_ARMOR_NAME_PATTERN.sub(self._armor_pattern_mapping, text)

    def _correct_base_abbreviation(self, text: str) -> str:
        return self.ABV_BASE_PATTERN.sub("<name>base</name>", text)

    def _correct_car_abbreviation(self, text: str) -> str:
        return self.ABV_CAR_PATTERN.sub("<name>car</name>", text)

    def _correct_watcher_abbreviation(self, text: str) -> str:
        return self.ABV_WATCHER_PATTERN.sub("<name>watcher</name>", text)

    def _correct_runes_abbreviation(self, text: str) -> str:
        def runes_abbreviation_mapping(match: Match):
            color = self.COLORS_MAP[match.group("color")]
            return f"<name>rune-{color}</name>"

        return self.ABV_RUNES_PATTERN.sub(runes_abbreviation_mapping, text)

    @staticmethod
    def _correct_mistakes(text):
        text = text.replace("\\</name>", "</name>")
        text = text.replace("<name>care</name>", "<name>car</name>")
        return text
