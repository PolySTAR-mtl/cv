from nptyping import Array

from polystar.models.roco_object import ArmorColor
from polystar.pipeline.classification.rule_based_classifier import RuleBasedClassifierABC


class RedBlueComparisonClassifier(RuleBasedClassifierABC):
    """A very simple model that compares the blue and red values obtained by the MeanChannels"""

    def predict_single(self, features: Array[float, float, float]) -> ArmorColor:
        return ArmorColor.RED if features[0] >= features[2] else ArmorColor.BLUE