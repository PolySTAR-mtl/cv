from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow import GraphDef, Session
from tensorflow.python.eager.wrap_function import WrappedFunction
from tensorflow.python.platform.gfile import GFile

from polystar.models.image import Image
from polystar.target_pipeline.detected_objects.objects_params import ObjectParams
from polystar.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC


@dataclass
class TFModelObjectsDetector(ObjectsDetectorABC):

    model_path: InitVar[Path]

    def __post_init__(self, model_path: Path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = GraphDef()
            with GFile(str(model_path / "frozen_inference_graph.pb"), "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

    def detect(self, image: Image) -> List[ObjectParams]:
        with self.graph.as_default(), Session(graph=self.graph) as session:
            image_np_expanded = np.expand_dims(image, axis=0)
            image_tensor = self.graph.get_tensor_by_name("image_tensor:0")
            boxes = self.graph.get_tensor_by_name("detection_boxes:0")
            scores = self.graph.get_tensor_by_name("detection_scores:0")
            classes = self.graph.get_tensor_by_name("detection_classes:0")
            num_detections = self.graph.get_tensor_by_name("num_detections:0")
            boxes, scores, classes, num_detections = session.run(
                [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded}
            )
            return _construct_objects_from_tf_results(boxes[0], scores[0], classes[0])


@dataclass
class TFV2ModelObjectsDetector(ObjectsDetectorABC):

    model: WrappedFunction

    def detect(self, image: Image) -> List[ObjectParams]:
        input_tensor = self._convert_image_to_input_tensor(image)
        output_dict = self._make_single_prediction(input_tensor)
        return _construct_objects_from_tf_results(
            output_dict["detection_boxes"], output_dict["detection_classes"], output_dict["detection_scores"]
        )

    @staticmethod
    def _convert_image_to_input_tensor(image: np.ndarray) -> tf.Tensor:
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        return input_tensor

    def _make_single_prediction(self, input_tensor: tf.Tensor) -> Dict[str, np.array]:
        output_dict: Dict[str, tf.Tensor] = self.model(input_tensor)
        return self._normalize_prediction(output_dict)

    @staticmethod
    def _normalize_prediction(output_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
        num_detections = int(output_dict.pop("num_detections"))
        output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
        output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)
        return output_dict


def _construct_objects_from_tf_results(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray
) -> List[ObjectParams]:
    return [
        ObjectParams(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax, score=score, object_class_id=object_class_id)
        for (ymin, xmin, ymax, xmax), object_class_id, score in zip(boxes, classes, scores)
        if score >= 0.1
    ]
