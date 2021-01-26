from typing import Callable

from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras.layers import Softmax, concatenate
from tensorflow.python.keras.losses import Loss, kullback_leibler_divergence
from tensorflow.python.ops.nn_ops import softmax

from polystar.pipeline.keras.model_preparator import KerasModelPreparator


class DistillationLoss(Loss):
    def __init__(self, temperature: float, n_classes: int):
        super().__init__(name="kd_loss")
        self.n_classes = n_classes
        self.temperature = temperature

    def call(self, y_true, y_pred):
        teacher_logits, student_logits = y_pred[:, : self.n_classes], y_pred[:, self.n_classes :]
        return kullback_leibler_divergence(
            softmax(teacher_logits / self.temperature, axis=1), softmax(student_logits / self.temperature, axis=1)
        )


class DistillationMetric:
    def __init__(self, metric: Callable, n_classes: int):
        self.n_classes = n_classes
        self.metric = metric
        self.__name__ = metric.__name__

    def __call__(self, y_true, y_pred):
        teacher_logits, student_logits = y_pred[:, : self.n_classes], y_pred[:, self.n_classes :]
        return self.metric(y_true, student_logits)


class Distiller(KerasModelPreparator):
    def __init__(
        self, teacher_model: Model, temperature: float,
    ):
        self.teacher_model = teacher_model
        self.temperature = temperature
        assert isinstance(teacher_model.layers[-1], Softmax)

    def prepare_model(self, model: Model) -> Model:
        assert isinstance(model.layers[-1], Softmax)

        self.teacher_model.trainable = False

        inputs = Input(shape=model.input.shape[1:])

        return Model(
            inputs=inputs,
            outputs=concatenate(
                [Sequential(self.teacher_model.layers[:-1])(inputs), Sequential(model.layers[:-1])(inputs)]
            ),
        )
