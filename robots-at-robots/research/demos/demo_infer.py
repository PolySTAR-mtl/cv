from polystar.common.models.label_map import LabelMap
from polystar.common.pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.plt_results_viewer import PltResultViewer
from polystar.robots_at_robots.dependency_injection import make_injector
from research.demos.utils import load_tf_model
from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset

if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    objects_detector = TFModelObjectsDetector(load_tf_model(), injector.get(LabelMap))
    filters = [ConfidenceObjectValidator(confidence_threshold=0.5)]

    with PltResultViewer("Demo of tf model") as viewer:
        for i, image in enumerate(SplitDataset(DJIROCODataset.CentralChina, Split.Test).images):
            objects = objects_detector.detect(image)
            for f in filters:
                objects = f.filter(objects, image)

            viewer.display_image_with_objects(image, objects)

            if i == 5:
                break
