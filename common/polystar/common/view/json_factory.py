from abc import abstractmethod, ABC
from typing import Generic, TypeVar, Dict, Any, NewType

Json = NewType("Json", Dict[str, Any])

T = TypeVar("T")


class JsonFactory(Generic[T], ABC):
    @staticmethod
    @abstractmethod
    def from_json(json: Json) -> T:
        pass

    @staticmethod
    @abstractmethod
    def to_json(example: T) -> Json:
        pass
