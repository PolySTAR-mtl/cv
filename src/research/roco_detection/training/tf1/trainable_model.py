from pathlib import Path
from typing import ClassVar

from google.protobuf.text_format import Merge
from object_detection.exporter import export_inference_graph
from object_detection.model_lib import create_estimator_and_inputs
from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig
from tensorflow_estimator.python.estimator.exporter import FinalExporter
from tensorflow_estimator.python.estimator.run_config import RunConfig
from tensorflow_estimator.python.estimator.training import EvalSpec, TrainSpec, train_and_evaluate

from polystar.utils.path import make_path
from research.constants import EVALUATION_DIR, PIPELINES_DIR


class TrainableModel:
    SAVE_CHECKPOINTS_STEPS: ClassVar[int] = 5  # 1_000
    EVAL_EVERY_SECS: ClassVar[int] = 30 * 60

    def __init__(self, config_path: Path, task: str, name: str):
        self.config_path = config_path
        self.name = name
        self.task = task
        self.training_path = make_path(EVALUATION_DIR / self.task / "tf1" / self.name)

    def train_and_export(self, nb_steps: int):
        self.launch_training(nb_steps=nb_steps)
        self.export()

    def launch_training(self, nb_steps: int):
        run_config = RunConfig(
            model_dir=str(self.training_path), save_checkpoints_steps=self.SAVE_CHECKPOINTS_STEPS, keep_checkpoint_max=2
        )
        train_and_eval_dict = create_estimator_and_inputs(
            run_config=run_config, pipeline_config_path=str(self.config_path)
        )
        estimator = train_and_eval_dict["estimator"]
        train_input_fn = train_and_eval_dict["train_input_fn"]
        eval_input_fns = train_and_eval_dict["eval_input_fns"]
        predict_input_fn = train_and_eval_dict["predict_input_fn"]

        train_spec = TrainSpec(train_input_fn, nb_steps)
        eval_spec = EvalSpec(
            name="0",
            input_fn=eval_input_fns[0],
            steps=None,
            exporters=FinalExporter(name="Servo", serving_input_receiver_fn=predict_input_fn),
            throttle_secs=self.EVAL_EVERY_SECS,
        )

        train_and_evaluate(estimator, train_spec, eval_spec)

        return self

    def export(self):
        pipeline_config = TrainEvalPipelineConfig()
        Merge(self.config_path.read_text(), pipeline_config)
        last_ckpt = max(self.training_path.glob("model.ckpt-*.meta"), key=_get_ckpt_number_from_file).with_suffix("")
        n_steps = last_ckpt.suffix.split("-")[-1]
        export_inference_graph(
            input_type="image_tensor",
            pipeline_config=pipeline_config,
            trained_checkpoint_prefix=str(last_ckpt),
            output_directory=str(PIPELINES_DIR / self.task / f"{self.name}__{n_steps}_steps"),
        )


def _get_ckpt_number_from_file(ckpt_file: Path):
    return int(ckpt_file.stem[len("model.ckpt-") :])
