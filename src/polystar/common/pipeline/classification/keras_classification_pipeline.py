from os.path import join
from typing import Callable, Dict, List, Sequence, Tuple, Union

from hypertune import HyperTune
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, TensorBoard
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.pipeline.keras.classifier import KerasClassifier
from polystar.common.pipeline.keras.cnn import make_cnn_model
from polystar.common.pipeline.keras.compilation_parameters import KerasCompilationParameters
from polystar.common.pipeline.keras.data_preparator import KerasDataPreparator
from polystar.common.pipeline.keras.distillation import DistillationLoss, DistillationMetric, Distiller
from polystar.common.pipeline.keras.trainer import KerasTrainer
from polystar.common.pipeline.keras.transfer_learning import make_transfer_learning_model
from polystar.common.pipeline.preprocessors.normalise import Normalise
from polystar.common.pipeline.preprocessors.resize import Resize


class KerasClassificationPipeline(ClassificationPipeline):
    @classmethod
    def from_model(cls, model: Model, trainer: KerasTrainer, input_shape: Tuple[int, int], name: str):
        return cls.from_pipes(
            [Resize(input_shape), Normalise(), KerasClassifier(model=model, trainer=trainer)], name=name
        )

    @classmethod
    def from_transfer_learning(
        cls,
        logs_dir: str,
        input_size: int,
        model_factory: Callable[..., Model],
        dropout: float,
        dense_size: int,
        lr: float,
        verbose: int = 0,
        name: str = None,
    ):
        input_shape = (input_size, input_size)
        name = name or f"{model_factory.__name__} ({input_size}) - lr {lr:.1e} - drop {dropout:.1%} - {dense_size}"
        return cls.from_model(
            model=make_transfer_learning_model(
                input_shape=input_shape,
                n_classes=cls.n_classes,
                model_factory=model_factory,
                dropout=dropout,
                dense_size=dense_size,
            ),
            trainer=make_classification_trainer(
                lr=lr, logs_dir=logs_dir, name=name, verbose=verbose, batch_size=32, steps_per_epoch=100
            ),
            name=name,
            input_shape=input_shape,
        )

    @classmethod
    def from_custom_cnn(
        cls,
        logs_dir: str,
        input_size: int,
        conv_blocks: Sequence[Sequence[int]],
        dropout: float,
        dense_size: int,
        lr: float,
        verbose: int = 0,
        name: str = None,
        batch_size: int = 32,
        steps_per_epoch: Union[str, int] = 100,
    ) -> ClassificationPipeline:
        name = name or (
            f"cnn - ({input_size}) - lr {lr:.1e} - drop {dropout:.1%} - "
            + " ".join("_".join(map(str, sizes)) for sizes in conv_blocks)
            + f" - {dense_size}"
        )
        input_shape = (input_size, input_size)
        return cls.from_model(
            make_cnn_model(
                input_shape=input_shape,
                conv_blocks=conv_blocks,
                dense_size=dense_size,
                output_size=cls.n_classes,
                dropout=dropout,
            ),
            trainer=make_classification_trainer(
                lr=lr,
                logs_dir=logs_dir,
                name=name,
                verbose=verbose,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
            ),
            name=name,
            input_shape=input_shape,
        )

    @classmethod
    def from_distillation(
        cls,
        teacher_pipeline: ClassificationPipeline,
        logs_dir: str,
        conv_blocks: Sequence[Sequence[int]],
        dropout: float,
        dense_size: int,
        lr: float,
        temperature: float,
        verbose: int = 0,
        name: str = None,
    ):
        input_shape: Tuple[int, int] = teacher_pipeline.named_steps["Resize"].size
        name = name or (
            f"distiled - temp {temperature:.1e}"
            f" - cnn - ({input_shape[0]}) - lr {lr:.1e} - drop {dropout:.1%} - "
            + " ".join("_".join(map(str, sizes)) for sizes in conv_blocks)
            + f" - {dense_size}"
        )
        return cls.from_model(
            model=make_cnn_model(
                input_shape, conv_blocks=conv_blocks, dense_size=dense_size, output_size=cls.n_classes, dropout=dropout,
            ),
            trainer=KerasTrainer(
                data_preparator=KerasDataPreparator(batch_size=32, steps=100),
                model_preparator=Distiller(temperature=temperature, teacher_model=teacher_pipeline.classifier.model),
                compilation_parameters=KerasCompilationParameters(
                    loss=DistillationLoss(temperature=temperature, n_classes=cls.n_classes),
                    metrics=[DistillationMetric(categorical_accuracy, n_classes=cls.n_classes)],
                    optimizer=Adam(lr),
                ),
                callbacks=make_classification_callbacks(join(logs_dir, name)),
                verbose=verbose,
            ),
            name=name,
            input_shape=input_shape,
        )


def make_classification_callbacks(log_dir: str) -> List[Callback]:
    return [
        EarlyStopping(verbose=0, patience=7, restore_best_weights=True, monitor="val_categorical_accuracy"),
        TensorBoard(log_dir=log_dir),
        # HyperTuneClassificationCallback(),
    ]


def make_classification_trainer(
    lr: float, logs_dir: str, name: str, verbose: int, batch_size: int, steps_per_epoch: Union[str, int]
) -> KerasTrainer:
    return KerasTrainer(
        compilation_parameters=KerasCompilationParameters(
            loss=CategoricalCrossentropy(from_logits=False), metrics=[categorical_accuracy], optimizer=Adam(lr)
        ),
        data_preparator=KerasDataPreparator(batch_size=batch_size, steps=steps_per_epoch),
        callbacks=make_classification_callbacks(join(logs_dir, name)),
        verbose=verbose,
    )


class HyperTuneClassificationCallback(Callback):
    def __init__(self):
        super().__init__()
        self.hpt = HyperTune()
        self.best_accuracy_epoch = (0, -1)

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        accuracy = logs["val_categorical_accuracy"]
        self._report(accuracy, epoch)
        self.best_accuracy_epoch = max(self.best_accuracy_epoch, (accuracy, epoch))

    def on_train_begin(self, logs=None):
        self.best_accuracy_epoch = (0, -1)

    def on_train_end(self, logs=None):
        self._report(*self.best_accuracy_epoch)

    def _report(self, accuracy: float, epoch: int):
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="val_accuracy", metric_value=accuracy, global_step=epoch
        )
