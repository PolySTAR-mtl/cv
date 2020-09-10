from typing import Generic, Iterator, List, Tuple, TypeVar

from polystar.common.models.image import Image

TargetT = TypeVar("TargetT")


class ImageDataset(Generic[TargetT]):
    def __init__(self, images: List[Image] = None, targets: List[TargetT] = None):
        self._targets = targets
        self._images = images
        self._check_consistency()

    def __iter__(self) -> Iterator[Tuple[Image, TargetT]]:
        return zip(self.images, self.targets)

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
        images, targets = zip(*self)
        self._images, self._targets = list(images), list(targets)
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
