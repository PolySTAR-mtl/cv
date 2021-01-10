from typing import Sequence, Tuple

from tensorflow.python.keras import Input, Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Softmax


def make_cnn_model(
    input_shape: Tuple[int, int],
    conv_blocks: Sequence[Sequence[int]],
    dense_size: int,
    output_size: int,
    dropout: float,
) -> Sequential:
    model = Sequential()
    model.add(Input((*input_shape, 3)))

    for conv_sizes in conv_blocks:
        for size in conv_sizes:
            model.add(Conv2D(size, (3, 3), activation="relu"))
        model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(dense_size))
    model.add(Dropout(dropout))
    model.add(Dense(output_size))
    model.add(Softmax())
    return model
