from enum import IntEnum, auto
from unittest import TestCase

from numpy import array_equal, asarray, ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.pipeline.classification.rule_based_classifier import RuleBasedClassifierABC
from polystar.common.pipeline.concat import Concat
from polystar.common.pipeline.pipe_abc import PipeABC


class Letter(IntEnum):
    A = auto()
    B = auto()
    Z = auto()


class FakeClassifier(RuleBasedClassifierABC):
    def predict_single(self, example: str) -> Letter:
        if example == "a":
            return Letter.A
        elif example == "b":
            return Letter.B
        else:
            return Letter.Z


class LetterPipeline(ClassificationPipeline):
    enum = Letter


class TestRuleBasedClassifier(TestCase):
    def setUp(self) -> None:
        self.pipeline = LetterPipeline.from_pipes([FakeClassifier()], Letter)

    def test_fit(self):
        self.pipeline.fit(list("bacbz"), [Letter.B, Letter.A, Letter.Z, Letter.B, Letter.Z])

    def test_n_classes(self):
        self.assertEqual(3, self.pipeline.classifier.n_classes)

    def test_predict(self):
        self.pipeline.classifier.n_classes = 3  # This is normally done during fitting
        self.assertEqual([Letter.A, Letter.A, Letter.Z, Letter.B, Letter.Z], self.pipeline.predict(list("aacbz")))

    def test_predict_proba(self):
        self.pipeline.classifier.n_classes = 3  # This is normally done during fitting
        array_equal(
            asarray([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]),
            self.pipeline.predict_proba(list("aacbz")),
        )

    def test_predict_proba_and_classes(self):
        self.pipeline.classifier.n_classes = 3  # This is normally done during fitting
        proba, classes = self.pipeline.predict_proba_and_classes(list("aacbz"))
        array_equal(asarray([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]]), proba)
        self.assertEqual([Letter.A, Letter.A, Letter.Z, Letter.B, Letter.Z], classes)


class StrToIntPipe(PipeABC[str, int]):
    def transform_single(self, example: str) -> int:
        return int(example)


class IntToArrayPipe(PipeABC[int, ndarray]):
    def transform_single(self, example: int) -> ndarray:
        return asarray([example])


class ReLuPipe(PipeABC[ndarray, ndarray]):
    def __init__(self, m: int):
        self.m = m

    def transform_single(self, example: ndarray) -> ndarray:
        return example * (example >= self.m)


class TestCV(TestCase):
    def setUp(self) -> None:
        self.pipeline = LetterPipeline.from_pipes(
            [StrToIntPipe(), IntToArrayPipe(), Concat.from_pipes([ReLuPipe(0), ReLuPipe(0)]), LogisticRegression()]
        )

    def test_cv(self):
        search = GridSearchCV(self.pipeline, {"Concat__ReLuPipe2__m": [0, 1, 2, 3, 4]}, cv=2)
        X = list("01234")
        y = [Letter.Z, Letter.A, Letter.B, Letter.Z, Letter.Z]
        search.fit(X * 10, y * 10)

        self.assertEqual(3, search.best_params_["Concat__ReLuPipe2__m"])
        self.assertEqual(y, search.best_estimator_.predict(X))
