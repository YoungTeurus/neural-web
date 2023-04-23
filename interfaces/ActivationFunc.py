from abc import ABC
from typing import Callable

FloatFunc = Callable[[float], float]


class ActivationFunc(ABC):
    f: FloatFunc
    fProizv: FloatFunc

    def f(self, x: float) -> float:
        raise NotImplementedError

    def fProizv(self, x: float) -> float:
        raise NotImplementedError
