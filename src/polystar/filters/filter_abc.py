from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Tuple, TypeVar

T = TypeVar("T")


class FilterABC(Generic[T], ABC):
    def filter(self, examples: List[T]) -> List[T]:
        return self.filter_with_siblings(examples)[0]

    def filter_with_siblings(self, examples: List[T], *siblings: List) -> Tuple[List[T], ...]:
        return self.split_with_siblings(examples, *siblings)[True]

    def split(self, examples: List[T]) -> Tuple[List[T], List[T]]:
        splits = self.split_with_siblings(examples)
        return splits[False][0], splits[True][0]

    def split_with_siblings(
        self, examples: List[T], *siblings: List
    ) -> Tuple[Tuple[List[T], ...], Tuple[List[T], ...]]:
        are_valid = self.validate(examples)

        if not any(are_valid):
            return (examples, *siblings), tuple([] for _ in range(len(siblings) + 1))
        elif all(are_valid):
            return tuple([] for _ in range(len(siblings) + 1)), (examples, *siblings)

        return (
            _filter_with_siblings_from_preds(are_valid, examples, *siblings, expected_value=False),
            _filter_with_siblings_from_preds(are_valid, examples, *siblings, expected_value=True),
        )

    def validate(self, examples: List[T]) -> List[bool]:
        return list(map(self.validate_single, examples))

    @abstractmethod
    def validate_single(self, example: T) -> bool:
        pass

    def __or__(self, other: "FilterABC") -> "FilterABC[T]":
        return UnionFilter(self, other)

    def __and__(self, other: "FilterABC") -> "FilterABC[T]":
        return IntersectionFilter(self, other)

    def __neg__(self) -> "FilterABC[T]":
        return NegationFilter(self)


class IntersectionFilter(FilterABC[T]):
    def __init__(self, *filters: FilterABC[T]):
        self.filters = filters
        assert self.filters

    def validate_single(self, example: T) -> bool:
        return all(f.validate_single(example) for f in self.filters)


class UnionFilter(FilterABC[T]):
    def __init__(self, *filters: FilterABC[T]):
        self.filters = filters
        assert self.filters

    def validate_single(self, example: T) -> bool:
        return any(f.validate_single(example) for f in self.filters)


class NegationFilter(FilterABC[T]):
    def __init__(self, base_filter: FilterABC[T]):
        self.base_filter = base_filter

    def validate_single(self, example: T) -> bool:
        return not self.base_filter.validate_single(example)


def _filter_with_siblings_from_preds(
    are_valid: List[bool], examples: List[T], *siblings: List, expected_value: bool = True
) -> Tuple[List[T], ...]:
    iterable_results = zip(
        *((ex, *s) for is_valid, ex, *s in zip(are_valid, examples, *siblings) if is_valid == expected_value)
    )
    return _format_res(iterable_results)


def _format_res(res: Tuple[Iterable[T]]) -> Tuple[List[T]]:
    return tuple(map(list, res))
