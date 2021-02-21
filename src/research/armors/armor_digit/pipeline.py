from polystar.models.roco_object import ArmorDigit
from polystar.pipeline.classification.classification_pipeline import ClassificationPipeline
#from polystar.pipeline.classification.keras_classification_pipeline import KerasClassificationPipeline
from polystar.utils.registry import registry


class ArmorDigitPipeline(ClassificationPipeline):
    enum = ArmorDigit



#@registry.register()
#class ArmorDigitKerasPipeline(ArmorDigitPipeline, KerasClassificationPipeline):
#    pass
