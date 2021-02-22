from dataclasses import dataclass

from polystar.target_pipeline.target_abc import SimpleTarget


@dataclass
class Command:
    id: int
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)


def make_target_command(target: SimpleTarget) -> Command:
    return Command(id=2, data=b"Y" + bytes(target))
