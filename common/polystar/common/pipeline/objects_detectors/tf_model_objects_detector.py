from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import tensorflow as tf

from polystar.common.models.image import Image
from polystar.common.models.label_map import LabelMap
from polystar.common.models.object import Object, ObjectType
from polystar.common.models.tf_model import TFModel
from polystar.common.pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC


@dataclass
class TFModelObjectsDetector(ObjectsDetectorABC):

    model: TFModel
    label_map: LabelMap

    def detect(self, image: Image) -> List[Object]:
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

    def _construct_objects_from_tf_results(self, image: Image, output_dict: Dict[str, np.ndarray]):
        image_height, image_width, *_ = image.shape
        objects: List[Object] = [
            self._construct_object_from_tf_result(box, class_id, image_height, image_width, score)
            for box, class_id, score in zip(
                output_dict["detection_boxes"], output_dict["detection_classes"], output_dict["detection_scores"]
            )
            if score >= 0.1
        ]
        return objects

    def _construct_object_from_tf_result(
        self, box: Tuple[float, float, float, float], class_id: int, image_height: int, image_width: int, score: float
    ):
        ymin, xmin, ymax, xmax = box
        object_type = ObjectType(self.label_map.name_of(class_id))
        return Object(
            type=object_type,
            confidence=score,
            x=int(xmin * image_width),
            y=int(ymin * image_height),
            w=int((xmax - xmin) * image_width),
            h=int((ymax - ymin) * image_height),
        )
