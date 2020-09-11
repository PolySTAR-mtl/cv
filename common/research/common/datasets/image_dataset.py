from typing import Generic, Iterable, Iterator, List, Tuple, TypeVar

from polystar.common.models.image import Image

TargetT = TypeVar("TargetT")


class ImageDataset(Generic[TargetT], Iterable[Tuple[Image, TargetT]]):
    def __init__(self, name: str, images: List[Image] = None, targets: List[TargetT] = None):
        self.name = name
        self._targets = targets
        self._images = images
        self._check_consistency()

    def __iter__(self) -> Iterator[Tuple[Image, TargetT]]:
        return zip(self.images, self.targets)

    def __str__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    __repr__ = __str__

    @property
    def images(self) -> List[Image]:
        self._load_data()
        return self._images

    @property
    def targets(self) -> List[TargetT]:
        self._load_data()
        return self._targets

    def _load_data(self):
        if self._is_loaded:
            return
        self._images, self._targets = map(list, zip(*self))
        self._check_consistency()

    def _check_consistency(self):
        assert self._is_loaded or self._has_custom_load
        if self._is_loaded:
            assert len(self.targets) == len(self.images)

    @property
    def _is_loaded(self) -> bool:
        return self._images is not None and self._targets is not None

    @property
    def _has_custom_load(self) -> bool:
        return not self.__iter__.__qualname__.startswith("ImageDataset")
