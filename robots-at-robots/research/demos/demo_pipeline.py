import cv2

from polystar.common.communication.print_target_sender import PrintTargetSender
from polystar.common.models.camera import Camera
from polystar.common.models.label_map import LabelMap
from polystar.common.models.object import ObjectType
from polystar.common.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.common.target_pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.target_pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.target_pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.target_pipeline.objects_validators.type_object_validator import TypeObjectValidator
from polystar.common.target_pipeline.target_pipeline import NoTargetFound
from polystar.common.target_pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.common.utils.tensorflow import patch_tf_v2
from polystar.common.view.plt_results_viewer import PltResultViewer
from polystar.robots_at_robots.dependency_injection import make_injector
from research.demos.utils import load_tf_model
from research_common.dataset.dji.dji_roco_datasets import DJIROCODataset
from research_common.dataset.split import Split
from research_common.dataset.split_dataset import SplitDataset
from research_common.dataset.twitch.twitch_roco_datasets import TwitchROCODataset

if __name__ == "__main__":
    patch_tf_v2()
    injector = make_injector()

    pipeline = DebugTargetPipeline(
        objects_detector=TFModelObjectsDetector(load_tf_model(), injector.get(LabelMap)),
        objects_validators=[ConfidenceObjectValidator(0.6), TypeObjectValidator(ObjectType.Armor)],
        object_selector=ClosestObjectSelector(),
        target_factory=RatioSimpleTargetFactory(injector.get(Camera), 300, 100),
        target_sender=PrintTargetSender(),
    )

    with PltResultViewer("Demo of tf model") as viewer:
        for dset in (TwitchROCODataset.TWITCH_470150052, SplitDataset(DJIROCODataset.CentralChina, Split.Test)):
            for i, image_path in enumerate(dset.image_paths):
                try:
                    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                    target = pipeline.predict_target(image)

                    viewer.new(image)
                    viewer.add_objects(pipeline.debug_info_.detected_objects, forced_color=(0.3, 0.3, 0.3))
                    viewer.add_objects(pipeline.debug_info_.validated_objects, forced_color=(0.6, 0.6, 0.6))
                    viewer.add_object(pipeline.debug_info_.selected_object)
                    viewer.display()
                except NoTargetFound:
                    pass

                if i == 5:
                    break
