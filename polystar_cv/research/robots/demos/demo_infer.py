from polystar.common.dependency_injection import make_injector
from polystar.common.models.label_map import LabelMap
from polystar.common.target_pipeline.detected_objects.detected_objects_factory import DetectedObjectFactory
from polystar.common.target_pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.target_pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.plt_results_viewer import PltResultViewer
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo
from research.robots.demos.utils import load_tf_model

if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    objects_detector = TFModelObjectsDetector(DetectedObjectFactory(injector.get(LabelMap), []), load_tf_model())
    filters = [ConfidenceObjectValidator(confidence_threshold=0.5)]

    with PltResultViewer("Demo of tf model") as viewer:
        for image, _, _ in ROCODatasetsZoo.DJI.CENTRAL_CHINA.to_images().cap(5):
            objects = objects_detector.detect(image)
            for f in filters:
                objects = f.filter(objects, image)

            viewer.display_image_with_objects(image, objects)
