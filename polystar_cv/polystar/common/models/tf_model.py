from typing import NewType

from tensorflow.python.eager.wrap_function import WrappedFunction

TFModel = NewType("TFModel", WrappedFunction)
