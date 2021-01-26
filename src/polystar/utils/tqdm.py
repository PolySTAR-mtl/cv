from typing import Iterable, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def smart_tqdm(
    it: Iterable[T], *args, desc: str = None, total: int = None, unit: str = "it", leave: bool = True, **kwargs
) -> Iterable[T]:
    return tqdm(it, *args, desc=desc, total=total, unit=unit, leave=leave, **kwargs)
