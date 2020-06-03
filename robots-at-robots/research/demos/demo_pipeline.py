import cv2

from polystar.common.communication.print_target_sender import PrintTargetSender
from polystar.common.image_pipeline.classifier_image_pipeline import ClassifierImagePipeline
from polystar.common.image_pipeline.image_featurizer.mean_rgb_channels_featurizer import MeanChannelsFeaturizer
from polystar.common.image_pipeline.models.red_blue_channels_comparison_model import RedBlueComparisonModel
from polystar.common.models.camera import Camera
from polystar.common.models.label_map import LabelMap
from polystar.common.target_pipeline.armors_descriptors.armors_color_descriptor import ArmorsColorDescriptor
from polystar.common.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.common.target_pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.target_pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.target_pipeline.objects_linker.simple_objects_linker import SimpleObjectsLinker
from polystar.common.target_pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.target_pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.common.target_pipeline.target_pipeline import NoTargetFoundException
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
        armors_descriptors=[
            ArmorsColorDescriptor(
                ClassifierImagePipeline(image_featurizer=MeanChannelsFeaturizer(), model=RedBlueComparisonModel())
            )
        ],
        objects_validators=[ConfidenceObjectValidator(0.6)],
        object_selector=ClosestObjectSelector(),
        target_factory=RatioSimpleTargetFactory(injector.get(Camera), 300, 100),
        target_sender=PrintTargetSender(),
        objects_linker=SimpleObjectsLinker(min_percentage_intersection=0.8),
    )

    with PltResultViewer("Demo of tf model") as viewer:
        for dset in (TwitchROCODataset.TWITCH_470150052, SplitDataset(DJIROCODataset.CentralChina, Split.Test)):
            for i, image_path in enumerate(dset.image_paths):
                try:
                    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
                    target = pipeline.predict_target(image)

                    viewer.new(image)
                    viewer.add_robots(pipeline.debug_info_.detected_robots, forced_color=(0.3, 0.3, 0.3))
                    viewer.add_robots(pipeline.debug_info_.validated_robots)
                    viewer.add_object(pipeline.debug_info_.selected_armor)
                    viewer.display()
                except NoTargetFoundException:
                    pass

                if i == 5:
                    break
