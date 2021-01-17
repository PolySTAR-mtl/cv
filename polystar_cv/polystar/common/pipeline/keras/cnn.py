from typing import Sequence, Tuple

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Softmax


def make_cnn_model(
    input_shape: Tuple[int, int],
    conv_blocks: Sequence[Sequence[int]],
    dense_size: int,
    output_size: int,
    dropout: float,
) -> Sequential:
    model = Sequential()
    model.add(Conv2D(conv_blocks[0][0], (3, 3), activation="relu", input_shape=(*input_shape, 3)))

    is_first = True
    for conv_sizes in conv_blocks:
        for size in conv_sizes:
            if is_first:
                is_first = False
                continue
            model.add(Conv2D(size, (3, 3), activation="relu"))
        model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(dense_size))
    model.add(Dropout(dropout))
    model.add(Dense(output_size))
    model.add(Softmax())
    return model
