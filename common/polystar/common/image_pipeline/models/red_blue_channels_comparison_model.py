from dataclasses import dataclass
from typing import List, Tuple

from polystar.common.image_pipeline.models.model_abc import ModelABC


@dataclass
class RedBlueComparisonModel(ModelABC):
    """A very simple model that compares the blue and red values obtained by the MeanChannelsFeaturizer"""

    red_channel_id: int = 0
    blue_channel_id: int = 2

    def predict(self, features: List[Tuple[float, float, float]]) -> List[str]:
        return [
            "Red" if feature[self.red_channel_id] >= feature[self.blue_channel_id] else "Blue" for feature in features
        ]

    def __str__(self) -> str:
        return "rb_comparison"
