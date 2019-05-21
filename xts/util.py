
from typing import Callable

import functools
import time


class timing(object):
    def __init__(self, name: str, verbose: bool = True) -> None:
        self.name = name
        self.verbose = verbose

    def __enter__(self) -> None:
        self.start = time.perf_counter()

        return self

    def __exit__(self, *args) -> None:
        self.end = time.perf_counter()
        self.total = self.end - self.start

        if self.verbose:
            timing.pp(self.name, self.start, self.end)

    @staticmethod
    def pp(title: str, start: float, end: float) -> None:
        dt = end - start

        if dt < 1:
            time_str = '{0:.3f}ms'.format(dt*1000)
        else:
            time_str = '{0:.3f}s'.format(dt)

        print('T', title, time_str, flush=True)

    @staticmethod
    def call(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timing(func.__name__):
                return func(*args, **kwargs)

        return wrapper
