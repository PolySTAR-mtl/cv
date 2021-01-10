from abc import ABC
from enum import Enum
from typing import Generic, List, Sequence, Union

from numpy.core._multiarray_umath import zeros

from polystar.common.pipeline.classification.classification_pipeline import EnumT
from polystar.common.pipeline.classification.classifier_abc import ClassifierABC
from polystar.common.pipeline.pipe_abc import IT


class RuleBasedClassifierABC(ClassifierABC, Generic[IT, EnumT], ABC):
    def predict_proba(self, examples: List[IT]) -> Sequence[float]:
        predictions = self.predict(examples)
        indices = [p.value - 1 if isinstance(p, Enum) else p for p in predictions]

        proba = zeros((len(examples), self.n_classes))
        proba[(range(len(predictions)), indices)] = 1

        return proba

    def predict(self, examples: List[IT]) -> List[Union[EnumT, int]]:
        return [self.predict_single(e) for e in examples]

    def predict_single(self, example: IT) -> Union[EnumT, int]:
        raise NotImplemented("You need to implement either predict or predict_single")
