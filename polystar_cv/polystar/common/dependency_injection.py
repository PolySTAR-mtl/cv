from dataclasses import dataclass
from typing import List

from injector import Injector, Module, multiprovider, provider, singleton
from numpy.core._multiarray_umath import deg2rad
from tensorflow.python import saved_model

from polystar.common.communication.print_target_sender import PrintTargetSender
from polystar.common.communication.target_sender_abc import TargetSenderABC
from polystar.common.constants import LABEL_MAP_PATH, MODELS_DIR
from polystar.common.models.camera import Camera
from polystar.common.models.label_map import LabelMap
from polystar.common.settings import Settings, settings
from polystar.common.target_pipeline.armors_descriptors.armors_color_descriptor import ArmorsColorDescriptor
from polystar.common.target_pipeline.armors_descriptors.armors_descriptor_abc import ArmorsDescriptorABC
from polystar.common.target_pipeline.detected_objects.detected_objects_factory import DetectedObjectFactory
from polystar.common.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.common.target_pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.common.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC
from polystar.common.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC
from polystar.common.target_pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector
from polystar.common.target_pipeline.objects_linker.objects_linker_abs import ObjectsLinkerABC
from polystar.common.target_pipeline.objects_linker.simple_objects_linker import SimpleObjectsLinker
from polystar.common.target_pipeline.objects_validators.confidence_object_validator import ConfidenceObjectValidator
from polystar.common.target_pipeline.objects_validators.objects_validator_abc import ObjectsValidatorABC
from polystar.common.target_pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.common.target_pipeline.target_factories.target_factory_abc import TargetFactoryABC
from research.robots.armor_color.pipeline import ArmorColorPipeline
from research.robots.armor_color.scripts.benchmark import MeanChannels, RedBlueComparisonClassifier


def make_injector() -> Injector:
    return Injector(modules=[CommonModule(settings)])


@dataclass
class CommonModule(Module):
    settings: Settings

    @provider
    @singleton
    def provide_camera(self) -> Camera:
        if settings.CAMERA == "RASPI_V2":
            return Camera(
                horizontal_fov=deg2rad(62.2),
                vertical_fov=deg2rad(48.8),
                pixel_size_m=1.12e-6,
                focal_m=3.04e-3,
                vertical_resolution=720,
                horizontal_resolution=1_280,
            )
        raise ValueError(f"Camera {settings.CAMERA} not recognized")

    @multiprovider
    @singleton
    def provide_label_map(self) -> LabelMap:
        return LabelMap.from_file(LABEL_MAP_PATH)

    @provider
    @singleton
    def provide_objects_detector(self, object_factory: DetectedObjectFactory) -> ObjectsDetectorABC:
        if self.settings.is_dev:
            tf_model = saved_model.load(str(MODELS_DIR / "robots" / settings.OBJECTS_DETECTION_MODEL / "saved_model"))
            return TFModelObjectsDetector(object_factory, tf_model.signatures["serving_default"])
        raise NotImplementedError()

    @multiprovider
    @singleton
    def provide_armor_descriptors(self) -> List[ArmorsDescriptorABC]:
        return [ArmorsColorDescriptor(ArmorColorPipeline.from_pipes([MeanChannels(), RedBlueComparisonClassifier()]))]

    @multiprovider
    @singleton
    def provide_objects_validators(self) -> List[ObjectsValidatorABC[DetectedRobot]]:
        return [ConfidenceObjectValidator(0.6)]

    @provider
    @singleton
    def provide_object_selector(self) -> ObjectSelectorABC:
        return ClosestObjectSelector()

    @provider
    @singleton
    def provide_target_factory(self, camera: Camera) -> TargetFactoryABC:
        return RatioSimpleTargetFactory(camera, 0.3, 0.1)

    @provider
    @singleton
    def provide_target_sender(self) -> TargetSenderABC:
        return PrintTargetSender()

    @provider
    @singleton
    def provide_objects_linker(self) -> ObjectsLinkerABC:
        return SimpleObjectsLinker(min_percentage_intersection=0.8)
