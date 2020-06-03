from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from polystar.common.image_pipeline.models.absolute_classifier_model_abc import AbsoluteClassifierModelABC


@dataclass
class RedBlueComparisonModel(AbsoluteClassifierModelABC):
    """A very simple model that compares the blue and red values obtained by the MeanChannelsFeaturizer"""

    red_channel_id: int = 0
    blue_channel_id: int = 2

    def __post_init__(self):
        self.labels_ = np.asarray(sorted(["Red", "Grey", "Blue"]))
        self.label2index_ = {label: i for i, label in enumerate(self.labels_)}

    def fit(self, features: List[Any], labels: List[Any]) -> "RedBlueComparisonModel":
        return self

    def predict(self, features: List[Tuple[float, float, float]]) -> List[str]:
        return [
            "Red" if feature[self.red_channel_id] >= feature[self.blue_channel_id] else "Blue" for feature in features
        ]

    def __str__(self) -> str:
        return "rb_comparison"
