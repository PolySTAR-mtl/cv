from datetime import datetime
from pathlib import Path
from typing import List

from google.protobuf.text_format import Merge, MessageToString
from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig
from object_detection.protos.preprocessor_pb2 import PreprocessingStep

from polystar.constants import LABEL_MAP_PATH, PROJECT_DIR
from research.roco_detection.training.tf1.records import Records
from research.roco_detection.training.tf1.trainable_model import TrainableModel

CONFIGS_DIR = PROJECT_DIR / "models/research/object_detection/samples/configs"


class ModelConfig:
    def __init__(self, pretrained_dir: Path, config_name: str):
        self.pretrained_dir = pretrained_dir
        self.config_name = config_name

    def configure(self, record: Records, data_augm: bool, height: int, width: int, n_classes: int) -> TrainableModel:
        config = self.read_config()

        self._update_config(config, record, data_augm, height, width, n_classes)

        config_path = self.pretrained_dir / "pipeline.config"

        config_path.write_text(MessageToString(config))

        size = f"{width}x{height}__" if width is not None else ""
        full_name = (
            f'{datetime.now().strftime("%y%m%d_%H%M%S")}__'
            f"{self.pretrained_dir.stem[:-16]}__"
            f"{size}"
            f'{"AUGM__" * data_augm}'
            f"{record.full_name}"
        )

        return TrainableModel(config_path, record.task_name, full_name)

    def _update_config(self, config, record: Records, data_augm: bool, height: int, width: int, n_classes: int):
        model_config = getattr(config.model, config.model.WhichOneof("model"))
        model_config.num_classes = n_classes
        _configure_input_shape(model_config, width, height)
        config.train_config.fine_tune_checkpoint = str(self.pretrained_dir / "model.ckpt")
        config.eval_config.max_evals = 0
        _configure_reader(config.train_input_reader, record.train)
        _configure_reader(config.eval_input_reader[0], record.val)
        if data_augm:
            _add_augmentations(
                config,
                [
                    # b"random_pixel_value_scale {}",
                    b"random_adjust_brightness {}",
                    b"random_adjust_contrast {}",
                    b"random_adjust_hue {}",
                    b"random_adjust_saturation {}",
                    b"random_jitter_boxes {}",
                ],
            )

    def read_config(self):
        config = TrainEvalPipelineConfig()
        Merge((CONFIGS_DIR / self.config_name).read_text(), config)
        return config


def _configure_reader(reader, record_path: Path):
    reader.label_map_path = str(LABEL_MAP_PATH)
    reader.tf_record_input_reader.input_path[0] = str(record_path)


def _configure_input_shape(model_config, width: int, height: int):
    if width is None:
        return
    shape_resizer = model_config.image_resizer.fixed_shape_resizer
    shape_resizer.width = width
    shape_resizer.height = height


def _add_augmentations(config, augmentations: List[bytes]):
    for augmentation in augmentations:
        config.train_config.data_augmentation_options.append(Merge(augmentation, PreprocessingStep()))
