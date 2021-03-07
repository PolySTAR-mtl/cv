from dataclasses import dataclass
from typing import List

from injector import Injector, Module, multiprovider, provider, singleton
from numpy.core._multiarray_umath import deg2rad
from serial import Serial

from polystar.communication.board_a import BoardA
from polystar.communication.cs_link_abc import CSLinkABC
from polystar.communication.screen import Screen
from polystar.constants import LABEL_MAP_PATH
from polystar.filters.filter_abc import FilterABC
from polystar.frame_generators.camera_frame_generator import CameraFrameGenerator, make_csi_camera_frame_generator
from polystar.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.models.camera import Camera
from polystar.models.label_map import LabelMap
from polystar.settings import Settings, settings
from polystar.target_pipeline.armors_descriptors.armors_descriptor_abc import ArmorsDescriptorABC
from polystar.target_pipeline.detected_objects.detected_robot import DetectedRobot
from polystar.target_pipeline.object_selectors.closest_object_selector import ClosestObjectSelector
from polystar.target_pipeline.object_selectors.object_selector_abc import ObjectSelectorABC
from polystar.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC
from polystar.target_pipeline.objects_filters.confidence_object_filter import RobotArmorConfidenceObjectsFilter
from polystar.target_pipeline.objects_linker.objects_linker_abs import ObjectsLinkerABC
from polystar.target_pipeline.objects_linker.simple_objects_linker import SimpleObjectsLinker
from polystar.target_pipeline.target_factories.ratio_simple_target_factory import RatioSimpleTargetFactory
from polystar.target_pipeline.target_factories.target_factory_abc import TargetFactoryABC
from research.common.constants import PIPELINES_DIR


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
    def provide_objects_detector(self) -> ObjectsDetectorABC:
        model_dir = PIPELINES_DIR / "roco-detection" / settings.OBJECTS_DETECTION_MODEL
        if self.settings.is_dev:
            from polystar.target_pipeline.objects_detectors.tf_model_objects_detector import TFModelObjectsDetector

            return TFModelObjectsDetector(model_dir)
        from polystar.target_pipeline.objects_detectors.trt_model_object_detector import TRTModelObjectsDetector

        return TRTModelObjectsDetector(model_dir)

    @multiprovider
    @singleton
    def provide_armor_descriptors(self) -> List[ArmorsDescriptorABC]:
        # TODO this needs to be mixed with filtering
        return [
            # ArmorsColorDescriptor(ArmorColorPipeline.from_pipes([MeanChannels(), RedBlueComparisonClassifier()])),
            # ArmorsDigitDescriptor(pkl_load(PIPELINES_DIR / "armor-digit" / settings.ARMOR_DIGIT_MODEL)),
        ]

    @provider
    @singleton
    def provide_robots_filter(self) -> FilterABC[DetectedRobot]:
        return RobotArmorConfidenceObjectsFilter(0.5)

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
    def provide_cs_link(self) -> CSLinkABC:
        if not self.settings.USE_UART:
            return Screen()
        return BoardA(Serial(settings.SERIAL_PORT, settings.BAUD_RATE))

    @provider
    @singleton
    def provide_objects_linker(self) -> ObjectsLinkerABC:
        return SimpleObjectsLinker(min_percentage_intersection=0.8)

    @provider
    def provide_webcam(self) -> FrameGeneratorABC:
        if self.settings.is_prod:
            return make_csi_camera_frame_generator(1_280, 720)
        return CameraFrameGenerator()
