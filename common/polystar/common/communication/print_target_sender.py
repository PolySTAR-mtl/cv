from polystar.common.communication.target_sender_abc import TargetSenderABC


class PrintTargetSender(TargetSenderABC):
    def _send_text(self, text: str):
        print(text)
