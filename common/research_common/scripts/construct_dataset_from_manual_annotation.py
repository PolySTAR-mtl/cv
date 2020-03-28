from os import remove
from pathlib import Path
from shutil import move, rmtree
from zipfile import ZipFile


def construct_dataset_from_manual_annotations(
    source_images_directory: Path, source_annotations_directory: Path, destination_directory: Path
):
    destination_images_directory = destination_directory / "image"
    destination_annotations_directory = destination_directory / "image_annotation"
    _unzip_all_in_directory(source_images_directory, destination_images_directory, "jpg")
    _unzip_all_in_directory(source_annotations_directory, destination_annotations_directory, "xml")

    names_of_annotated = {annotation_file.stem for annotation_file in destination_annotations_directory.glob("*.xml")}
    for image_file in destination_images_directory.glob("*.jpg"):
        if image_file.stem not in names_of_annotated:
            remove(str(image_file))


def _unzip_all_in_directory(source_directory: Path, destination_directory: Path, extension: str):
    for zip_path in source_directory.glob("*.zip"):
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination_directory)
    for file_path in destination_directory.glob(f"**/*.{extension}"):
        move(str(file_path), str(destination_directory / file_path.name))
    for directory in destination_directory.glob("*"):
        if directory.is_dir():
            rmtree(str(directory))
