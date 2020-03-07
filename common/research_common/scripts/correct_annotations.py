import re
from pathlib import Path

from research_common.constants import TWITCH_DSET_DIR


class AnnotationFileCorrector:
    FINAL_ARMOR_NAME_PATTERN = re.compile(r"<name>armor-(?P<color>\w{2,4})-(?P<num>\d)</name>")
    ABV_ARMOR_NAME_PATTERN = re.compile(r"<name>a(?P<color>\w)(?P<num>\d)</name>")
    ABV_RUNES_PATTERN = re.compile(r"<name>r(?P<color>\w)</name>")
    ABV_BASE_PATTERN = re.compile(r"<name>b</name>")
    ABV_WATCHER_PATTERN = re.compile(r"<name>w</name>")
    ABV_CAR_PATTERN = re.compile(r"<name>c</name>")

    COLORS_MAP = {"r": "red", "red": "red", "b": "blue", "blue": "blue", "g": "gray", "grey": "gray", "gray": "gray"}

    def correct_annotations_in_directory(self, directory: Path):
        for annotation_path in directory.glob("*.xml"):
            self.correct_annotation_file(annotation_path)

    def correct_annotation_file(self, annotation_file: Path):
        text = self._save_previous_content(annotation_file)
        text = self._correct_annotation_text(text)
        annotation_file.write_text(text)

    def _correct_annotation_text(self, text: str) -> str:
        text = self._correct_abbreviations(text)
        text = self._correct_armor_format(text)
        return text

    @staticmethod
    def _save_previous_content(annotation_file: Path) -> str:
        previous_content = annotation_file.read_text()
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

    def _armor_pattern_mapping(self, match: re.Match):
        color = self.COLORS_MAP[match.group("color")]
        num = match.group("num")
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
        def runes_abbreviation_mapping(match: re.Match):
            color = self.COLORS_MAP[match.group("color")]
            return f"<name>rune-{color}</name>"

        return self.ABV_RUNES_PATTERN.sub(runes_abbreviation_mapping, text)


if __name__ == "__main__":

    corrector = AnnotationFileCorrector()

    annotation_dir = TWITCH_DSET_DIR / "robots-views-annotations" / "chunk_005"

    corrector.correct_annotations_in_directory(annotation_dir)
