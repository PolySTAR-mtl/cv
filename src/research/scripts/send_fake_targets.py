from cmath import pi
from time import sleep

from injector import inject

from polystar.communication.cs_link_abc import CSLinkABC
from polystar.dependency_injection import make_injector
from polystar.target_pipeline.target_abc import SimpleTarget


@inject
def send_fake_targets_in_loop(cs_link: CSLinkABC):
    for i in range(60):
        cs_link.send_target(SimpleTarget(d=i / 100, phi=0, theta=pi / 2))
        sleep(1 / 60)


if __name__ == "__main__":
    make_injector().call_with_injection(send_fake_targets_in_loop)
