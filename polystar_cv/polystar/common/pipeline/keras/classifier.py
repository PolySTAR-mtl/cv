from copy import copy
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Sequence

from numpy import asarray
from tensorflow import Graph, Session
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from polystar.common.models.image import Image
from polystar.common.pipeline.classification.classifier_abc import ClassifierABC
from polystar.common.pipeline.keras.trainer import KerasTrainer
from polystar.common.settings import settings
from polystar.common.utils.registry import registry


@registry.register()
class KerasClassifier(ClassifierABC):
    def __init__(self, model: Model, trainer: KerasTrainer):
        self.model = model
        self.trainer: Optional[KerasTrainer] = trainer

    def fit(self, images: List[Image], labels: List[int], validation_size: int) -> "KerasClassifier":
        assert self.trainable, "You can't train an un-pickled classifier"
        images = asarray(images)
        labels = to_categorical(asarray(labels), self.n_classes)
        train_images, train_labels = images[:-validation_size], labels[:-validation_size]
        val_images, val_labels = images[-validation_size:], labels[-validation_size:]

        self.trainer.train(self.model, train_images, train_labels, val_images, val_labels)

        return self

    def predict_proba(self, examples: List[Image]) -> Sequence[float]:
        if settings.is_prod:  # FIXME
            with self.graph.as_default(), self.session.as_default():
                return self.model.predict(asarray(examples))
        return self.model.predict(asarray(examples))

    def __getstate__(self) -> Dict:
        with NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            self.model.save(fd.name, overwrite=True, include_optimizer=False)
            model_str = fd.read()
        state = copy(self.__dict__)
        state.pop("model")
        state.pop("trainer")
        return {**state, "model_str": model_str}

    def __setstate__(self, state: Dict):
        self.__dict__.update(state)
        self.graph = Graph()
        with NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
            fd.write(state.pop("model_str"))
            fd.flush()
            with self.graph.as_default():
                self.session = Session(graph=self.graph)
                with self.session.as_default():
                    self.model = load_model(fd.name, compile=False)
        self.trainer = None

    @property
    def trainable(self) -> bool:
        return self.trainer is not None
