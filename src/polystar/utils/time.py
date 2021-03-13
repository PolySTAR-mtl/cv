from datetime import datetime
from functools import wraps
from typing import Callable


def create_time_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def time_it(f: Callable):
    @wraps(f)
    def _f(*args, **kwargs):
        t = datetime.now()
        rv = f(*args, **kwargs)
        print(f"{f.__name__} took {datetime.now() - t}")
        return rv

    return _f
