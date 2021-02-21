from contextlib import suppress
from time import sleep

from injector import inject

from polystar.communication.target_sender_abc import TargetSenderABC
from polystar.dependency_injection import make_injector
from polystar.target_pipeline.target_abc import SimpleTarget


@inject
def send_fake_targets_in_loop(target_sender: TargetSenderABC, fps: float):
    target = SimpleTarget(theta=0.25, phi=-0.35, d=5.3)

    with suppress(KeyboardInterrupt):
        while True:
            target_sender.send(target)
            sleep(1 / fps)


if __name__ == "__main__":
    make_injector().call_with_injection(send_fake_targets_in_loop, kwargs=dict(fps=10))
