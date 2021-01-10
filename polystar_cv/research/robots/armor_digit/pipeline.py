from polystar.common.models.object import ArmorDigit
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.pipeline.classification.keras_classification_pipeline import KerasClassificationPipeline
from polystar.common.utils.registry import registry


class ArmorDigitPipeline(ClassificationPipeline):
    enum = ArmorDigit


@registry.register()
class ArmorDigitKerasPipeline(ArmorDigitPipeline, KerasClassificationPipeline):
    pass
