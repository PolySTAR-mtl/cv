from typing import Callable, Tuple

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine import InputLayer
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Softmax
from tensorflow.python.keras.models import Model


def make_transfer_learning_model(
    input_shape: Tuple[int, int], n_classes: int, model_factory: Callable[..., Model], dropout: float, dense_size: int,
) -> Sequential:
    input_shape = (*input_shape, 3)
    base_model: Model = model_factory(weights="imagenet", input_shape=input_shape, include_top=False)

    return Sequential(
        [
            InputLayer(input_shape),
            base_model,
            Flatten(),
            Dense(dense_size, activation="relu"),
            Dropout(dropout),
            Dense(n_classes),
            Softmax(),
        ]
    )
