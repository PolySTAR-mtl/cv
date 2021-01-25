import json
from abc import ABC, abstractmethod

from polystar.common.target_pipeline.target_abc import TargetABC


class TargetSenderABC(ABC):
    def send(self, target: TargetABC):
        self._send_text(json.dumps(target.to_json()))

    @abstractmethod
    def _send_text(self, text: str):
        pass
