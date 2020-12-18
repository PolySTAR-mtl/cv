import logging
import warnings
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from keras_preprocessing.image import ImageDataGenerator
from numpy import asarray
from tensorflow_core.python.keras import Input, Model, Sequential
from tensorflow_core.python.keras.applications.vgg16 import VGG16
from tensorflow_core.python.keras.applications.xception import Xception
from tensorflow_core.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow_core.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from polystar.common.image_pipeline.preprocessors.normalise import Normalise
from polystar.common.image_pipeline.preprocessors.resize import Resize
from polystar.common.models.image import Image
from polystar.common.models.object import ArmorDigit
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.pipeline.classification.classifier_abc import ClassifierABC
from polystar.common.pipeline.classification.random_model import RandomClassifier
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots_at_robots.armor_digit.armor_digit_benchmarker import make_armor_digit_benchmarker
from research.robots_at_robots.armor_digit.armor_digit_dataset import make_armor_digit_dataset_generator
from research.robots_at_robots.evaluation.benchmark import Benchmarker


class ArmorDigitPipeline(ClassificationPipeline):
    enum = ArmorDigit


class KerasClassifier(ClassifierABC):
    def __init__(self, model: Model, optimizer, logs_dir: Path, with_data_augmentation: bool, batch_size: int = 32):
        self.batch_size = batch_size
        self.logs_dir = logs_dir
        self.with_data_augmentation = with_data_augmentation
        self.model = model
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    @property
    def train_data_gen(self) -> ImageDataGenerator:
        if not self.with_data_augmentation:
            return ImageDataGenerator()
        return ImageDataGenerator(rotation_range=45, zoom_range=[0.8, 1])  # brightness_range=[0.7, 1.4]

    def fit(self, images: List[Image], labels: List[int], validation_size: int) -> "KerasClassifier":
        images = asarray(images)
        labels = to_categorical(asarray(labels), 5)  # FIXME
        train_images, train_labels = images[:-validation_size], labels[:-validation_size]
        val_images, val_labels = images[-validation_size:], labels[-validation_size:]

        train_generator = self.train_data_gen.flow(train_images, train_labels, batch_size=self.batch_size, shuffle=True)

        self.model.fit(
            x=train_generator,
            steps_per_epoch=len(train_images) / self.batch_size,
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


class CNN(Sequential):
    def __init__(
        self, input_size: Tuple[int, int], conv_blocks: Sequence[Sequence[int]], dense_size: int, output_size: int,
    ):
        super().__init__()
        self.add(Input((*input_size, 3)))

        for conv_sizes in conv_blocks:
            for size in conv_sizes:
                self.add(Conv2D(size, (3, 3), activation="relu"))
            self.add(MaxPooling2D())

        self.add(Flatten())
        self.add(Dense(dense_size))
        self.add(Dropout(0.5))
        self.add(Dense(output_size, activation="softmax"))


def make_digits_cnn_pipeline(
    input_size: int, conv_blocks: Sequence[Sequence[int]], report_dir: Path, with_data_augmentation: bool, lr: float
) -> ArmorDigitPipeline:
    name = (
        f"cnn - ({input_size}) - lr {lr} - "
        + " ".join("_".join(map(str, sizes)) for sizes in conv_blocks)
        + (" - with_data_augm" * with_data_augmentation)
    )
    input_size = (input_size, input_size)
    return ArmorDigitPipeline.from_pipes(
        [
            Resize(input_size),
            Normalise(),
            KerasClassifier(
                CNN(input_size=input_size, conv_blocks=conv_blocks, dense_size=128, output_size=5),
                logs_dir=report_dir / name,
                with_data_augmentation=with_data_augmentation,
                optimizer=SGD(lr, momentum=0.9),
            ),
        ],
        name=name,
    )


class TransferLearning(Sequential):
    def __init__(self, input_size: Tuple[int, int], n_classes: int, model_factory: Callable[..., Model]):
        input_shape = (*input_size, 3)
        base_model: Model = model_factory(weights="imagenet", input_shape=input_shape, include_top=False)

        super().__init__(
            [
                Input(input_shape),
                base_model,
                Flatten(),
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(n_classes, activation="softmax"),
            ]
        )


def make_tl_pipeline(
    report_dir: Path, input_size: int, with_data_augmentation: bool, lr: float, model_factory: Callable[..., Model]
):
    name = f"{model_factory.__name__} ({input_size}) - lr {lr}" + (" - with_data_augm" * with_data_augmentation)
    input_size = (input_size, input_size)
    return ArmorDigitPipeline.from_pipes(
        [
            Resize(input_size),
            Normalise(),
            KerasClassifier(
                model=TransferLearning(input_size=input_size, n_classes=5, model_factory=model_factory),  # FIXME
                optimizer=Adam(lr),  # FIXME
                logs_dir=report_dir,
                with_data_augmentation=with_data_augmentation,
            ),
        ],
        name=name,
    )


def make_vgg16_pipeline(
    report_dir: Path, input_size: int, with_data_augmentation: bool, lr: float
) -> ArmorDigitPipeline:
    return make_tl_pipeline(
        model_factory=VGG16,
        input_size=input_size,
        with_data_augmentation=with_data_augmentation,
        lr=lr,
        report_dir=report_dir,
    )


def make_xception_pipeline(
    report_dir: Path, input_size: int, with_data_augmentation: bool, lr: float
) -> ArmorDigitPipeline:
    return make_tl_pipeline(
        model_factory=Xception,
        input_size=input_size,
        with_data_augmentation=with_data_augmentation,
        lr=lr,
        report_dir=report_dir,
    )


def make_default_digit_benchmarker(exp_name: str) -> Benchmarker:
    return make_armor_digit_benchmarker(
        train_digit_datasets=[
            # Only the start of the dataset is cleaned as of now
            make_armor_digit_dataset_generator()
            .from_roco_dataset(ROCODatasetsZoo.DJI.FINAL)
            .to_file_images()
            .cap(
                (1009 - 117)
                + (1000 - 86)
                + (1000 - 121)
                + (1000 - 138)
                + (1000 - 137)
                + (1000 - 154)
                + (1000 - 180)
                + (1000 - 160)
                + (1000 - 193)
                + (1000 - 80)
            )
            .build()
        ],
        train_roco_datasets=[
            # ROCODatasetsZoo.DJI.CENTRAL_CHINA,
            # ROCODatasetsZoo.DJI.FINAL,
            # ROCODatasetsZoo.DJI.NORTH_CHINA,
            # ROCODatasetsZoo.DJI.SOUTH_CHINA,
            ROCODatasetsZoo.TWITCH.T470150052,
            ROCODatasetsZoo.TWITCH.T470149568,
            ROCODatasetsZoo.TWITCH.T470151286,
        ],
        validation_roco_datasets=[ROCODatasetsZoo.TWITCH.T470152289],
        test_roco_datasets=[
            ROCODatasetsZoo.TWITCH.T470152838,
            ROCODatasetsZoo.TWITCH.T470153081,
            ROCODatasetsZoo.TWITCH.T470158483,
            ROCODatasetsZoo.TWITCH.T470152730,
        ],
        experiment_name=exp_name,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel("INFO")
    logging.getLogger("tensorflow").setLevel("ERROR")
    warnings.filterwarnings("ignore")

    _benchmarker = make_default_digit_benchmarker("xception")

    _report_dir = _benchmarker.reporter.report_dir

    _random_pipeline = ArmorDigitPipeline.from_pipes([RandomClassifier()], name="random")
    _cnn_pipelines = [
        make_digits_cnn_pipeline(
            32, ((32, 32), (64, 64)), _report_dir, with_data_augmentation=with_data_augmentation, lr=lr,
        )
        for with_data_augmentation in [False]
        for lr in [2.5e-2, 1.6e-2, 1e-2, 6.3e-3, 4e-4]
    ]
    # cnn_pipelines = [
    #     make_digits_cnn_pipeline(
    #         64, ((32,), (64, 64), (64, 64)), reporter.report_dir, with_data_augmentation=True, lr=lr
    #     )
    #     for with_data_augmentation in [True, False]
    #     for lr in (5.6e-2, 3.1e-2, 1.8e-2, 1e-2, 5.6e-3, 3.1e-3, 1.8e-3, 1e-3)
    # ]

    vgg16_pipelines = [
        make_vgg16_pipeline(_report_dir, input_size=32, with_data_augmentation=False, lr=lr)
        for lr in (1e-5, 5e-4, 2e-4, 1e-4, 5e-3)
    ]

    xception_pipelines = [
        make_xception_pipeline(_report_dir, input_size=71, with_data_augmentation=False, lr=lr) for lr in (2e-4, 5e-5)
    ]

    logging.info(f"Run `tensorboard --logdir={_report_dir}` for realtime logs")

    _benchmarker.benchmark([_random_pipeline] + xception_pipelines)
