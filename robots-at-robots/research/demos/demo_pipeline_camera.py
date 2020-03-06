import cv2

from polystar.common.models.camera import Camera
from polystar.common.models.object import ObjectType
from polystar.common.pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.pipeline.objects_validators.type_object_validator import TypeObjectValidator
from polystar.common.pipeline.pipeline import Pipeline
from polystar.common.pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.robots_at_robots.dependency_injection import make_injector
from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset

if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    # objects_detector = injector.get(TFModelObjectsDetector)
    # filters = [ConfidenceObjectValidator(confidence_threshold=0.5)]

    cap = cv2.VideoCapture(0)

    for i, image_path in enumerate(SplitDataset(DJIROCODataset.CentralChina, Split.Test).image_paths):
        ret, image = cap.read()
        # objects = objects_detector.detect(image)

        # Display the resulting frame
        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
