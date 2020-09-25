from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple
from unittest import TestCase

from numpy import asarray, float32
from numpy.testing import assert_array_almost_equal

from polystar.common.models.image import save_image
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder


class TestDirectoryROCODataset(TestCase):
    def test_file(self):
        with TemporaryDirectory() as dataset_dir:
            dataset_dir = Path(dataset_dir)

            annotation = ROCOAnnotation(objects=[], has_rune=False, w=160, h=90)

            images_dir, annotations_dir = self._setup_dir(dataset_dir)

            (annotations_dir / "frame_1.xml").write_text(annotation.to_xml())
            (images_dir / "frame_1.jpg").write_text("")

            dataset = ROCODatasetBuilder(dataset_dir, "fake").build_lazy()
            self.assertEqual([(images_dir / "frame_1.jpg", annotation, "frame_1")], list(dataset))

    def test_image(self):
        with TemporaryDirectory() as dataset_dir:
            dataset_dir = Path(dataset_dir)

            annotation = ROCOAnnotation(objects=[], has_rune=False, w=160, h=90)
            image = asarray([[[250, 0, 0], [250, 0, 0]], [[250, 0, 0], [250, 0, 0]]]).astype(float32)

            images_dir, annotations_dir = self._setup_dir(dataset_dir)

            (annotations_dir / "frame_1.xml").write_text(annotation.to_xml())
            save_image(image, images_dir / "frame_1.jpg")

            dataset = ROCODatasetBuilder(dataset_dir, "fake").to_images().build()
            self.assertEqual([annotation], list(dataset.targets))
            images = list(dataset.examples)
            self.assertEqual(1, len(images))
            assert_array_almost_equal(image / 256, images[0] / 256, decimal=2)  # jpeg precision

    def _setup_dir(self, dataset_dir: Path) -> Tuple[Path, Path]:
        images_dir = dataset_dir / "image"
        annotations_dir = dataset_dir / "image_annotation"

        annotations_dir.mkdir()
        images_dir.mkdir()

        return images_dir, annotations_dir
