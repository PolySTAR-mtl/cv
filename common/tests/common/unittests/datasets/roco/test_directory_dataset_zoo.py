from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from research.common.datasets.roco.directory_roco_dataset import \
    DirectoryROCODataset
from research.common.datasets.roco.roco_annotation import ROCOAnnotation


class TestDirectoryROCODataset(TestCase):
    def test_lazy_targets(self):
        with TemporaryDirectory() as dataset_dir:
            dataset = DirectoryROCODataset(Path(dataset_dir), "fake")
            dataset.annotations_dir_path.mkdir()

            annotation = ROCOAnnotation("frame_1", objects=[], has_rune=False, w=160, h=90)
            (dataset.annotations_dir_path / "frame_1.xml").write_text(annotation.to_xml())

            self.assertEqual([annotation], dataset)
