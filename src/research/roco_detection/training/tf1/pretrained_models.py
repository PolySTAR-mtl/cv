import logging
import tarfile
from contextlib import closing
from enum import Enum
from urllib.request import urlretrieve

from polystar.utils.path import make_path
from research.constants import EVALUATION_DIR
from research.roco_detection.training.tf1.model_config import ModelConfig
from research.roco_detection.training.tf1.records import Records
from research.roco_detection.training.tf1.trainable_model import TrainableModel

logger = logging.getLogger(__name__)

PRETRAINED_MODELS_DIR = make_path(EVALUATION_DIR / "pretrained")


class PretrainedModels(Enum):
    SSD_MOBILENET_V2 = ("ssd_mobilenet_v2_coco_2018_03_29", "ssd_mobilenet_v2_coco.config")
    FASTER_RCNN_INCEPTION_V2 = ("faster_rcnn_inception_v2_coco_2018_01_28", "faster_rcnn_inception_v2_pets.config")
    RFCN_RESENET101 = ("rfcn_resnet101_coco_2018_01_28", "rfcn_resnet101_pets.config")

    def __init__(self, model_name: str, config_name: str):
        self.config_name = config_name
        self.model_name = model_name
        self.pretrained_dir = PRETRAINED_MODELS_DIR / model_name
        self.config = ModelConfig(self.pretrained_dir, self.config_name)

    def setup(
        self,
        record: Records,
        task: str,
        data_augm: bool = False,
        height: int = None,
        width: int = None,
        n_classes: int = 5,
    ) -> TrainableModel:
        self._download()
        return self.config.configure(record, task, data_augm=data_augm, height=height, width=width, n_classes=n_classes)

    def _download(self):
        if self.pretrained_dir.exists():
            logger.info(f"model {self.model_name} already downloaded")
            return

        zip_file = f"{self.pretrained_dir}.tar.gz"

        # fetch
        urlretrieve(
            f"http://download.tensorflow.org/models/object_detection/{self.model_name}.tar.gz", zip_file,
        )

        # unzip
        with closing(tarfile.open(zip_file)) as tar:
            tar.extractall(PRETRAINED_MODELS_DIR)
