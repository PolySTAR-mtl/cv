import cv2

from polystar.common.models.camera import Camera
from polystar.common.models.label_map import LabelMap
from polystar.common.models.object import ObjectType
from polystar.common.pipeline.debug_pipeline import DebugPipeline
from polystar.common.pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.pipeline.objects_validators.type_object_validator import TypeObjectValidator
from polystar.common.pipeline.pipeline import NoTargetFound
from polystar.common.pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.plt_results_viewer import PltResultViewer
from polystar.robots_at_robots.dependency_injection import make_injector
from research.demos.utils import load_tf_model
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset
from research_common.dataset.twitch.twitch_roco_datasets import TwitchROCODataset

if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    pipeline = DebugPipeline(
        objects_detector=TFModelObjectsDetector(load_tf_model(), injector.get(LabelMap)),
        objects_validators=[ConfidenceObjectValidator(0.6), TypeObjectValidator(ObjectType.Armor)],
        object_selector=ClosestObjectSelector(),
        target_factory=RatioSimpleTargetFactory(injector.get(Camera), 300, 100),
    )

    with PltResultViewer("Demo of tf model") as viewer:
        for i, image_path in enumerate(SplitDataset(TwitchROCODataset.TWITCH_470150052, Split.Test).image_paths):
            try:
                image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                target = pipeline.predict_target(image)

                viewer.new(image)
                viewer.add_objects(pipeline.debug_info_.validated_objects, forced_color=(0.3, 0.3, 0.3))
                viewer.add_object(pipeline.debug_info_.selected_object)
                viewer.display()
            except NoTargetFound:
                pass

            if i == 5:
                break
