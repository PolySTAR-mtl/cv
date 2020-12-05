from collections import Counter
from typing import List

import numpy as np
from numpy.random.mtrand import choice

from polystar.common.pipeline.classification.rule_based_classifier import RuleBasedClassifierABC


class RandomClassifier(RuleBasedClassifierABC):
    def predict(self, examples: np.ndarray) -> List[int]:
        return choice(range(self.n_classes), size=len(examples), replace=True, p=self.weights_)

    def fit(self, examples: List, label_indices: List[int]) -> "RandomClassifier":
        indices2counts = Counter(label_indices)
        self.weights_ = [indices2counts[i] / len(label_indices) for i in range(self.n_classes)]
        return self
