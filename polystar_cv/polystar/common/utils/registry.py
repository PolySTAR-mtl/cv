from itertools import chain
from typing import Dict, Sequence, Type

from polystar.common.utils.singleton import Singleton


class Registry(Dict[str, Type], Singleton):
    def register(self, previous_names: Sequence[str] = ()):
        def decorator(class_: Type):
            for name in chain((class_.__name__,), previous_names):
                assert name not in self, f"{name} is already registered"
                self[name] = class_
            return class_

        return decorator


registry = Registry()
