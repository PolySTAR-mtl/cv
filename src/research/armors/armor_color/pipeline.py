from polystar.models.roco_object import ArmorColor
from polystar.pipeline.classification.classification_pipeline import ClassificationPipeline
#from polystar.pipeline.classification.keras_classification_pipeline import KerasClassificationPipeline


class ArmorColorPipeline(ClassificationPipeline):
    enum = ArmorColor



#class ArmorColorKerasPipeline(ArmorColorPipeline, KerasClassificationPipeline):
#    pass
