from typing import Any, Tuple, Union

from numpy.core._multiarray_umath import ndarray
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


class KerasDataPreparator:
    def __init__(self, batch_size: int, steps: Union[str, int]):
        self.steps = steps
        self.batch_size = batch_size

    def prepare_training_data(self, images: ndarray, labels: ndarray) -> Tuple[Any, int]:
        train_datagen = ImageDataGenerator()
        steps = self.steps if isinstance(self.steps, int) else len(images) / self.batch_size
        return train_datagen.flow(images, labels, batch_size=self.batch_size, shuffle=True), steps
