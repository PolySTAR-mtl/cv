from abc import ABC, abstractmethod

from polystar.target_pipeline.target_abc import SimpleTarget


class TargetSenderABC(ABC):
    @abstractmethod
    def send(self, target: SimpleTarget):
        pass
