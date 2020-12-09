import logging
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import seaborn as sns
from cv2.cv2 import resize
from keras_preprocessing.image import ImageDataGenerator
from numpy import asarray
from tensorflow_core.python.keras import Input, Model, Sequential
from tensorflow_core.python.keras.applications.vgg16 import VGG16
from tensorflow_core.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow_core.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from polystar.common.models.image import Image
from polystar.common.models.object import ArmorDigit
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.pipeline.classification.classifier_abc import ClassifierABC
from polystar.common.pipeline.classification.random_model import RandomClassifier
from polystar.common.pipeline.pipe_abc import PipeABC
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.armor_digit.armor_digit_pipeline_reporter_factory import (
    ArmorDigitPipelineReporterFactory,
)


class ArmorDigitPipeline(ClassificationPipeline):
    enum = ArmorDigit


class KerasClassifier(ClassifierABC):
    def __init__(self, model: Model, optimizer, logs_dir: Path, with_data_augmentation: bool):
        self.logs_dir = logs_dir
        self.with_data_augmentation = with_data_augmentation
        self.model = model
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    @property
    def train_data_gen(self) -> ImageDataGenerator:
        if not self.with_data_augmentation:
            return ImageDataGenerator()
        return ImageDataGenerator(rotation_range=45, zoom_range=[0.8, 1])  # brightness_range=[0.7, 1.4]

    def fit(self, images: List[Image], labels: List[int]) -> "KerasClassifier":
        n_val: int = 540  # FIXME
        images = asarray(images)
        labels = to_categorical(asarray(labels), 5)  # FIXME
        train_images, train_labels = images[:-n_val], labels[:-n_val]
        val_images, val_labels = images[-n_val:], labels[-n_val:]

        batch_size = 32  # FIXME
        train_generator = self.train_data_gen.flow(train_images, train_labels, batch_size)

        self.model.fit(
            x=train_generator,
            steps_per_epoch=len(train_images) / batch_size,
            validation_data=(val_images, val_labels),
            epochs=300,
            callbacks=[
                EarlyStopping(verbose=0, patience=15, restore_best_weights=True),
                TensorBoard(log_dir=self.logs_dir, histogram_freq=4, write_graph=True, write_images=False),
            ],
            verbose=0,
        )
        return self

    def predict_proba(self, examples: List[Image]) -> Sequence[float]:
        return self.model.predict_proba(asarray(examples))


class CNN(KerasClassifier):
    def __init__(
        self,
        input_size: Tuple[int, int],
        conv_blocks: Sequence[Sequence[int]],
        dense_size: int,
        output_size: int,
        logs_dir: Path,
        with_data_augmentation: bool,
        lr: float,
    ):
        model = Sequential()
        model.add(Input((*input_size, 3)))

        for conv_sizes in conv_blocks:
            for size in conv_sizes:
                model.add(Conv2D(size, (3, 3), activation="relu"))
            model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(dense_size))
        model.add(Dropout(0.5))
        model.add(Dense(output_size, activation="softmax"))

        super().__init__(
            model, optimizer=SGD(lr=lr, momentum=0.9), logs_dir=logs_dir, with_data_augmentation=with_data_augmentation,
        )


class Resize(PipeABC):
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def transform_single(self, image: Image) -> Image:
        return resize(image, self.size)


class Normalise(PipeABC):
    def transform_single(self, image: Image) -> Image:
        return image / 255


def make_digits_cnn_pipeline(
    input_size: int, conv_blocks: Sequence[Sequence[int]], report_dir: Path, with_data_augmentation: bool, lr: float
) -> ArmorDigitPipeline:
    name = (
        f"cnn - ({input_size}) - lr {lr} - "
        + " / ".join("_".join(map(str, sizes)) for sizes in conv_blocks)
        + (" - with_data_augm" * with_data_augmentation)
    )
    input_size = (input_size, input_size)
    return ArmorDigitPipeline.from_pipes(
        [
            Resize(input_size),
            Normalise(),
            CNN(
                input_size=input_size,
                conv_blocks=conv_blocks,
                dense_size=128,
                output_size=5,
                logs_dir=report_dir / name,
                with_data_augmentation=with_data_augmentation,
                lr=lr,
            ),
        ],
        name=name,
    )


class TransferLearning(KerasClassifier):
    def __init__(
        self, logs_dir: Path, input_size: Tuple[int, int], n_classes: int, with_data_augmentation: bool, lr: float,
    ):
        input_shape = (*input_size, 3)
        vgg16: Model = VGG16(weights="imagenet", input_shape=input_shape, include_top=False)

        model = Sequential(
            [
                Input(input_shape),
                vgg16,
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(n_classes, activation="softmax"),
            ]
        )
        super().__init__(model, Adam(lr), logs_dir, with_data_augmentation)


def make_vgg16_pipeline(
    report_dir: Path, input_size: int, with_data_augmentation: bool, lr: float
) -> ArmorDigitPipeline:
    name = f"vgg16 ({input_size}) - lr {lr}" + (" - with_data_augm" * with_data_augmentation)
    input_size = (input_size, input_size)
    return ArmorDigitPipeline.from_pipes(
        [
            Resize(input_size),
            Normalise(),
            TransferLearning(
                logs_dir=report_dir / name,
                input_size=input_size,
                n_classes=5,
                with_data_augmentation=with_data_augmentation,
                lr=lr,
            ),
        ],
        name=name,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")

    sns.set_style()

    reporter = ArmorDigitPipelineReporterFactory.from_roco_datasets(
        train_roco_datasets=[
            # ROCODatasetsZoo.DJI.CENTRAL_CHINA,
            # ROCODatasetsZoo.DJI.FINAL,
            # ROCODatasetsZoo.DJI.NORTH_CHINA,
            # ROCODatasetsZoo.DJI.SOUTH_CHINA,
            ROCODatasetsZoo.TWITCH.T470150052,
            ROCODatasetsZoo.TWITCH.T470149568,
            ROCODatasetsZoo.TWITCH.T470151286,
            ROCODatasetsZoo.TWITCH.T470152289,
        ],
        test_roco_datasets=[
            #
            ROCODatasetsZoo.TWITCH.T470152838,
            ROCODatasetsZoo.TWITCH.T470153081,
            ROCODatasetsZoo.TWITCH.T470158483,
            ROCODatasetsZoo.TWITCH.T470152730,
        ],
        experiment_name="data_augm",
    )

    random_pipeline = ArmorDigitPipeline.from_pipes([RandomClassifier()], name="random")

    cnn_pipelines = [
        make_digits_cnn_pipeline(32, ((32, 32), (64, 64)), reporter.report_dir, with_data_augmentation=True, lr=lr)
        for lr in (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4)
    ] + [
        make_digits_cnn_pipeline(
            64, ((32,), (64, 64), (64, 64)), reporter.report_dir, with_data_augmentation=False, lr=lr
        )
        for lr in (5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3)
    ]

    vgg16_pipelines = [
        make_vgg16_pipeline(reporter.report_dir, input_size=32, with_data_augmentation=True, lr=lr)
        for lr in (1e-5, 5e-4, 2e-4, 1e-4, 5e-3)
    ]

    logging.info(f"Run `tensorboard --logdir={reporter.report_dir}` for realtime logs")

    reporter.report([random_pipeline, *cnn_pipelines, *vgg16_pipelines])
