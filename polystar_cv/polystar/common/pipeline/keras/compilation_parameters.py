from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.metrics import Metric
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


@dataclass
class KerasCompilationParameters:
    optimizer: Union[str, OptimizerV2]
    loss: Union[str, Callable, Loss]
    metrics: List[Union[str, Callable, Metric]]
    loss_weights: Optional[Dict[str, float]] = None
