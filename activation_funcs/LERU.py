from interfaces.ActivationFunc import ActivationFunc


class LERU(ActivationFunc):
    def f(self, x: float) -> float:
        if x < 0:
            return x / 10
        return x

    def fProizv(self, x: float) -> float:
        if x < 0:
            return 1/10
        return 1