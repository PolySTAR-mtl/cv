from polystar.common.models.object import ArmorColor
from polystar.common.pipeline.classification.classification_pipeline import ClassificationPipeline
from polystar.common.pipeline.classification.keras_classification_pipeline import KerasClassificationPipeline


class ArmorColorPipeline(ClassificationPipeline):
    enum = ArmorColor


class ArmorColorKerasPipeline(ArmorColorPipeline, KerasClassificationPipeline):
    pass
