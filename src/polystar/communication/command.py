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


NO_TARGET_COMMAND = Command(id=2, data=b"N\x00\x00\x00\x00\x00\x00")
