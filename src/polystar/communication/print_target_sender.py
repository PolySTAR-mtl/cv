from polystar.communication.target_sender_abc import TargetSenderABC
from polystar.target_pipeline.target_abc import SimpleTarget


class PrintTargetSender(TargetSenderABC):
    def send(self, target: SimpleTarget):
        print(f"{target} ({bytes(target)})")
