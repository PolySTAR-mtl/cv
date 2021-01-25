from polystar.common.filters.filter_abc import FilterABC
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC
from research.common.datasets.roco.roco_annotation import ROCOAnnotation


class ROCOAnnotationObjectFilter(FilterABC):
    def __init__(self, object_validator: ObjectsValidatorABC):
        self.object_validator = object_validator

    def validate_single(self, annotation: ROCOAnnotation) -> bool:
        return any(self.object_validator.validate(annotation.objects, None))
