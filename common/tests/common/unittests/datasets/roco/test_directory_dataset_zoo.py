from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from numpy import asarray, float32
from numpy.testing import assert_array_almost_equal

from polystar.common.models.image import save_image
from research.common.datasets.roco.directory_roco_dataset import DirectoryROCODataset
from research.common.datasets_v3.roco.roco_annotation import ROCOAnnotation


class TestDirectoryROCODataset(TestCase):
    def test_targets(self):
        with TemporaryDirectory() as dataset_dir:
            dataset = DirectoryROCODataset(Path(dataset_dir), "fake")

            annotation = ROCOAnnotation("frame_1", objects=[], has_rune=False, w=160, h=90)

            dataset.annotations_dir.mkdir()
            dataset.images_dir.mkdir()
            (dataset.annotations_dir / "frame_1.xml").write_text(annotation.to_xml())
            (dataset.images_dir / "frame_1.jpg").write_text("")

            self.assertEqual([annotation], list(dataset.targets))
            self.assertEqual([dataset.images_dir / "frame_1.jpg"], list(dataset.examples))

    def test_open(self):
        with TemporaryDirectory() as dataset_dir:
            dataset = DirectoryROCODataset(Path(dataset_dir), "fake")

            annotation = ROCOAnnotation("frame_1", objects=[], has_rune=False, w=160, h=90)
            image = asarray([[[250, 0, 0], [250, 0, 0]], [[250, 0, 0], [250, 0, 0]]]).astype(float32)

            dataset.annotations_dir.mkdir()
            dataset.images_dir.mkdir()
            (dataset.annotations_dir / "frame_1.xml").write_text(annotation.to_xml())
            save_image(image, dataset.images_dir / "frame_1.jpg")

            image_dataset = dataset.open()

            self.assertEqual([annotation], list(image_dataset.targets))
            images = list(image_dataset.examples)
            self.assertEqual(1, len(images))
            assert_array_almost_equal(image / 256, images[0] / 256, decimal=2)  # jpeg precision
