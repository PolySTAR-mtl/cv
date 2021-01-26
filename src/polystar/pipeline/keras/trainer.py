from dataclasses import dataclass, field
from typing import List

from numpy.core._multiarray_umath import ndarray
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import Model

from polystar.pipeline.keras.compilation_parameters import KerasCompilationParameters
from polystar.pipeline.keras.data_preparator import KerasDataPreparator
from polystar.pipeline.keras.model_preparator import KerasModelPreparator


@dataclass
class KerasTrainer:
    compilation_parameters: KerasCompilationParameters
    callbacks: List[Callback]
    data_preparator: KerasDataPreparator
    model_preparator: KerasModelPreparator = field(default_factory=KerasModelPreparator)
    max_epochs: int = 300
    verbose: int = 0

    def train(
        self,
        model: Model,
        train_images: ndarray,
        train_labels: ndarray,
        validation_images: ndarray,
        validation_labels: ndarray,
    ):
        model = self.model_preparator.prepare_model(model)
        model.compile(**self.compilation_parameters.__dict__)
        train_data, steps = self.data_preparator.prepare_training_data(train_images, train_labels)

        model.fit(
            x=train_data,
            validation_data=(validation_images, validation_labels),
            steps_per_epoch=steps,
            epochs=self.max_epochs,
            callbacks=self.callbacks,
            verbose=self.verbose,
        )
