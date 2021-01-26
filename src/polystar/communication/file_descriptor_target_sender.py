from os import fdopen

from polystar.communication.target_sender_abc import TargetSenderABC


class FileDescriptorTargetSender(TargetSenderABC):
    def __init__(self, output_fd: int):
        self.output_fds = fdopen(int(output_fd), "w", buffering=1)

    def _send_text(self, text: str):
        self.output_fds.write(text + "\n")
