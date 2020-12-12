from itertools import islice
from typing import Iterator, Optional, Tuple

from research.common.datasets.lazy_dataset import ExampleT, LazyDataset, TargetT


class SliceDataset(LazyDataset):
    def __init__(
        self,
        source: LazyDataset[ExampleT, TargetT],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ):
        super().__init__(source.name)
        self.step = step
        self.stop = stop
        self.start = start
        self.source = source

    def __iter__(self) -> Iterator[Tuple[ExampleT, TargetT]]:
        return islice(self.source, self.start, self.stop, self.step)
