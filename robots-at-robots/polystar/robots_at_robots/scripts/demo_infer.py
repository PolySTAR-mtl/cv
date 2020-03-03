from polystar.common.pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.display_image_annotation import display_image_with_objects
from polystar.robots_at_robots.dependency_injection import make_injector
from research_common.dataset.roco.roco_datasets import ROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset

if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    objects_detector = injector.get(TFModelObjectsDetector)
    filters = [ConfidenceObjectValidator(confidence_threshold=0.5)]

    for i, image in enumerate(SplitDataset(ROCODataset.CentralChina, Split.Test).images):
        objects = objects_detector.detect(image)
        for f in filters:
            objects = f.filter(objects, image)

        display_image_with_objects(image, objects)
        if i == 0:
            break
