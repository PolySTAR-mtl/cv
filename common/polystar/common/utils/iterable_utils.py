from collections import defaultdict
from itertools import chain
from typing import Callable, Dict, Iterable, List, TypeVar

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
