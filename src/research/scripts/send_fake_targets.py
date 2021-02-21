import logging
from contextlib import suppress
from time import sleep
from typing import Optional

from dataclasses import dataclass
from injector import inject
from serial import Serial

from polystar.communication.target_sender_abc import TargetSenderABC
from polystar.dependency_injection import make_injector
from polystar.target_pipeline.target_abc import SimpleTarget


@dataclass
class Command:
    id: int
    data: bytes


@inject
def send_fake_targets_in_loop(target_sender: TargetSenderABC, serial: Serial, fps: float):
    target = SimpleTarget(theta=0.25, phi=-0.35, d=5.3)

    with suppress(KeyboardInterrupt):
        while True:
            target_sender.send(target)

            c = get_command(serial)
            if c:
                print(c)

            sleep(1 / fps)


def get_command(serial: Serial) -> Optional[Command]:
    if not serial.in_waiting:
        return

    starting_byte = serial.read()
    if starting_byte != b"\xfc":
        logging.warning(f"Wrong starting byte: {starting_byte}")
        return get_command(serial)

    command_id = int.from_bytes(serial.read(size=2), byteorder="big")
    size = int.from_bytes(serial.read(), byteorder="big")
    data = serial.read(size)

    return Command(command_id, data)


if __name__ == "__main__":
    make_injector().call_with_injection(send_fake_targets_in_loop, kwargs=dict(fps=40))
