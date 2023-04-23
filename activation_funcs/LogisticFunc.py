from math import exp

from interfaces.ActivationFunc import ActivationFunc


class Logistic(ActivationFunc):
    def f(self, x: float) -> float:
        return 1/(1+exp(-x))

    def fProizv(self, x: float) -> float:
        f = self.f(x)
        return f * (1 - f)
