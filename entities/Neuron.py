from Utils import multAndSum, generateWeights
from interfaces.ActivationFunc import ActivationFunc


class Neuron:
    weights: list[float]
    activationFunc: ActivationFunc

    def __init__(self, weights: list[float], activationFunc: ActivationFunc):
        self.weights = weights
        self.activationFunc = activationFunc

    @classmethod
    def createNew(cls, shape: int, activationFunc: ActivationFunc, startWeights: list[float] | None = None) -> 'Neuron':
        if shape < 0:
            raise RuntimeError("Can't create neuron with shape 0 or less")

        if startWeights is not None:
            if shape != len(startWeights):
                raise RuntimeError("Can't create neuron with shape {} using weights {}".format(shape, startWeights))
            weights = startWeights
        else:
            weights = generateWeights(shape)

        return Neuron(weights, activationFunc)

    def getNet(self, _input: list[float]) -> float:
        neuronShape = len(self.weights)
        inputShape = len(_input)
        if neuronShape != inputShape:
            raise RuntimeError(
                "Shape of input ({}) is not equals to shape of neuron ({})".format(inputShape, neuronShape))

        return multAndSum(self.weights, _input)

    def activate(self, _input: list[float]) -> float:
        net = self.getNet(_input)
        result = self.activationFunc.f(net)
        return result

    def __str__(self) -> str:
        return str(self.weights)
