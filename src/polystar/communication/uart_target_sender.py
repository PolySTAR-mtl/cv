from serial import Serial

from polystar.communication.target_sender_abc import TargetSenderABC
from polystar.target_pipeline.target_abc import SimpleTarget


class UartTargetSender(TargetSenderABC):
    START: bytes = b"\xfc\x00\x02\x07Y"

    def __init__(self, serial_port: Serial):
        self.serial_port = serial_port

    def send(self, target: SimpleTarget):
        self.serial_port.write(self.START + bytes(target))

    def __del__(self):
        self.serial_port.close()


if __name__ == "__main__":
    print(bytes(SimpleTarget(theta=0.25, phi=-0.35, d=5.3)))
