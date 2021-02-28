from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, Iterable, List, TypeVar

from more_itertools import ilen


def smart_len(it: Iterable) -> int:
    try:
        return len(it)
    except AttributeError:
        return ilen(it)


T = TypeVar("T")


def flatten(it: Iterable[Iterable[T]]) -> List[T]:
    return list(chain.from_iterable(it))


U = TypeVar("U")


def group_by(it: Iterable[T], key: Callable[[T], U]) -> Dict[U, List[T]]:
    rv = defaultdict(list)
    for item in it:
        rv[key(item)].append(item)
    return rv


def chunk(it: Iterable[T], batch_size: float) -> Iterable[List[T]]:
    batch = []
    for el in it:
        batch.append(el)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def apply(f: Callable[[T], Any], it: Iterable[T]):
    for el in it:
        f(el)
