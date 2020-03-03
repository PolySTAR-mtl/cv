import re

from research_common.constants import TWITCH_DSET_DIR

ANNOTATION_DIR = TWITCH_DSET_DIR / 'image_annotation'

file_path = ANNOTATION_DIR.glob('*.xml')

pattern = re.compile(r"<name>armor-(?P<color>\w{2,4})-(?P<num>\d)</name>")


def pattern_detect(match: re.Match):
    color = match.group("color")
    num = match.group("num")
    return f"<name>armor</name> <armor_class>{num}</armor_class> <armor_color>{color}</armor_color>"


output_dir = ANNOTATION_DIR / 'output'


if __name__ == '__main__':

    output_dir.mkdir(exist_ok=True)

    for file_path_input in file_path:
        content = file_path_input.read_text()
        content = pattern.sub(pattern_detect, content)
        file_path_output = output_dir / file_path_input.name
        file_path_output.write_text(content)

