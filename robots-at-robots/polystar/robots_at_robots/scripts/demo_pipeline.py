import cv2

from polystar.common.models.camera import Camera
from polystar.common.models.object import ObjectType
from polystar.common.pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.pipeline.objects_validators.type_object_validator import TypeObjectValidator
from polystar.common.pipeline.pipeline import Pipeline
from polystar.common.pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.robots_at_robots.dependency_injection import make_injector
from research_common.dataset.roco.roco_datasets import ROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset


if __name__ == "__main__":
    injector = make_injector()

    pipeline = Pipeline(
        objects_detector=injector.get(TFModelObjectsDetector),
        objects_validators=[ConfidenceObjectValidator(0.6), TypeObjectValidator(ObjectType.Armor)],
        object_selector=ClosestObjectSelector(),
        target_factory=RatioSimpleTargetFactory(injector.get(Camera), 300, 100),
    )

    for i, image_path in enumerate(SplitDataset(ROCODataset.CentralChina, Split.Test).image_paths):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        print(pipeline.process(image))
        if i == 0:
            break
