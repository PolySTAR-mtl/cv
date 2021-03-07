import ctypes
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pycuda.autoinit  # This is needed for initializing CUDA driver
import pycuda.driver as cuda
import tensorrt as trt

from polystar.constants import RESOURCES_DIR
from polystar.models.image import Image
from polystar.target_pipeline.detected_objects.objects_params import ObjectParams
from polystar.target_pipeline.objects_detectors.objects_detector_abc import ObjectsDetectorABC


@dataclass
class TRTModelObjectsDetector(ObjectsDetectorABC):
    model_dir: InitVar[Path]

    def __post_init__(self, model_dir: Path):
        self.trt_model = TRTModel(model_dir / "trt_model.bin", (300, 300))

    def detect(self, image: Image) -> List[ObjectParams]:
        results = self.trt_model(image)
        return _construct_objects_from_trt_results(results)


def _construct_objects_from_trt_results(results: np.ndarray) -> List[ObjectParams]:
    return [
        ObjectParams(
            ymin=float(ymin),
            xmin=float(xmin),
            ymax=float(ymax),
            xmax=float(xmax),
            score=float(score),
            object_class_id=int(object_class_id),
        )
        for (_, object_class_id, score, xmin, ymin, xmax, ymax) in results
        if object_class_id == 4 and score >= 0.1
    ]


class TRTModel:
    def __init__(self, trt_model_path: Path, input_size: Tuple[int, int]):
        self.input_size = input_size

        self.cuda_ctx = cuda.Device(0).make_context()

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine(trt_model_path)

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def __call__(self, img: Image) -> np.ndarray:
        img_resized = self._preprocess_image(img)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        return self.host_outputs[0].reshape((-1, 7))

    # Processing

    def _preprocess_image(self, img: Image) -> Image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = (2.0 / 255.0) * img - 1.0
        return img

    # Initialization

    def _load_plugins(self):
        if trt.__version__[0] < "7":
            ctypes.CDLL(str(RESOURCES_DIR / "nano/libflattenconcat.so"))
        trt.init_libnvinfer_plugins(self.trt_logger, "")

    def _load_engine(self, trt_model_path: Path):
        with trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(trt_model_path.read_bytes())

    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    # Delete

    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs
        self.cuda_ctx.pop()
