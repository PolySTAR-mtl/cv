from typing import NewType

from tensorflow_core.python.eager.wrap_function import WrappedFunction

TFModel = NewType("TFModel", WrappedFunction)
