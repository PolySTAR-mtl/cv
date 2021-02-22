from contextlib import suppress
from time import sleep

from injector import inject

from polystar.communication.command import make_target_command
from polystar.communication.cs_link_abc import CSLinkABC
from polystar.dependency_injection import make_injector
from polystar.target_pipeline.target_abc import SimpleTarget


@inject
def send_fake_targets_in_loop(cs_link: CSLinkABC, fps: float):
    target = SimpleTarget(theta=0.25, phi=-0.35, d=5.3)

    with suppress(KeyboardInterrupt):
        while True:
            cs_link.send_command(make_target_command(target))

            for c in cs_link.read_commands():
                print(c)

            sleep(1 / fps)


if __name__ == "__main__":
    make_injector().call_with_injection(send_fake_targets_in_loop, kwargs=dict(fps=0.5))
