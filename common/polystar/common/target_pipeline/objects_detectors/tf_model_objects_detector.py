from typing import List, Dict

import numpy as np
import tensorflow as tf
from dataclasses import dataclass

from polystar.common.models.image import Image
from polystar.common.models.label_map import LabelMap
from polystar.common.models.tf_model import TFModel
from polystar.common.target_pipeline.detected_objects.detected_object import DetectedObject
from polystar.common.target_pipeline.detected_objects.detected_objects_factory import DetectedObjectFactory
from polystar.common.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC


@dataclass
class TFModelObjectsDetector(ObjectsDetectorABC):

    model: TFModel
    label_map: LabelMap

    def detect(self, image: Image) -> List[DetectedObject]:
        input_tensor = self._convert_image_to_input_tensor(image)
        output_dict = self._make_single_prediction(input_tensor)
        return self._construct_objects_from_tf_results(image, output_dict)

    @staticmethod
    def _convert_image_to_input_tensor(image: np.ndarray) -> tf.Tensor:
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        return input_tensor

    def _make_single_prediction(self, input_tensor: tf.Tensor) -> Dict[str, np.array]:
        output_dict: Dict[str, tf.Tensor] = self.model(input_tensor)  # typing is correct despite PyCharm's saying
        return self._normalize_prediction(output_dict)

    @staticmethod
    def _normalize_prediction(output_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
        num_detections = int(output_dict.pop("num_detections"))
        output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
        output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)
        return output_dict

    def _construct_objects_from_tf_results(
        self, image: Image, output_dict: Dict[str, np.ndarray]
    ) -> List[DetectedObject]:
        objects_factory = DetectedObjectFactory(image, self.label_map)
        objects = [
            objects_factory.from_relative_positions(
                ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax, score=score, object_class_id=object_class_id
            )
            for (ymin, xmin, ymax, xmax), object_class_id, score in zip(
                output_dict["detection_boxes"], output_dict["detection_classes"], output_dict["detection_scores"]
            )
            if score >= 0.1
        ]
        return objects
