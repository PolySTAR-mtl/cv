from typing import Iterable

from more_itertools import ilen


def smart_len(it: Iterable) -> int:
    try:
        return len(it)
    except AttributeError:
        return ilen(it)
