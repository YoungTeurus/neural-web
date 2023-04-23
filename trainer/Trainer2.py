from typing import Callable

from entities.NeuralWeb import NeuralWeb, NeuralWebInput, NeuralWebOutput, LayersNetsAndOuts

Data = tuple[NeuralWebInput, NeuralWebOutput]
TrainData = list[Data]
Gradients = list[list[float]]
NewWeights = list[list[list[float]]]

DEFAULT_SPEED = 0.5
DEFAULT_MAX_EPOCH = 1000
DEFAULT_MINIMAL_EPOCH_ERROR_DIFF = 0.01


class TrainOptions:
    speed: float
    maxEpoch: int
    minimalEpochErrorDiff: float

    def __init__(self, speed: float, maxEpoch: int, errorDecreaseStop: float):
        self.speed = speed
        self.maxEpoch = maxEpoch
        self.minimalEpochErrorDiff = errorDecreaseStop

    @classmethod
    def default(cls) -> 'TrainOptions':
        return TrainOptions(speed=DEFAULT_SPEED, maxEpoch=DEFAULT_MAX_EPOCH,
                            errorDecreaseStop=DEFAULT_MINIMAL_EPOCH_ERROR_DIFF)


def train(nw: NeuralWeb, trainData: TrainData, trainOptions: TrainOptions = None, detailedLogs: bool = False,
          doOnEachEpoch: Callable[[NeuralWeb], None] | None = None) -> None:
    assert (len(trainData) > 0)

    speed = trainOptions.speed

    training: bool = True
    epoch: int = 1
    lastEpochError: float | None = None
    while training:
        print("Start {} epoch".format(epoch))

        thisEpochError = 0

        for data in trainData:
            _in, _out = data

            realOut, netsAndOuts = nw.process(_in)

            gradients: Gradients = [[] for _ in range(len(nw.layers))]
            newWs: NewWeights = [[] for _ in range(len(nw.layers))]

            isOutputLayer = True
            for _layerIndex in range(len(nw.layers), 0, -1):
                layerIndex = _layerIndex - 1
                layer = nw.layers[layerIndex]
                for neuronIndex, neuron in enumerate(layer):
                    # Градиент этого нейрона:
                    gradient = getGradient(isOutputLayer, layerIndex, neuronIndex, nw, netsAndOuts, gradients, realOut,
                                           _out)
                    gradients[layerIndex].append(gradient)

                    newNeuronWs = []
                    for wIndex, wOld in enumerate(neuron.weights):
                        # Выход нейрона предыдущего слоя, идущий в этот нейрон по весу wIndex
                        out = getOut(netsAndOuts, layerIndex, wIndex, _in)

                        wNew = wOld - speed * gradient * out
                        newNeuronWs.append(wNew)
                    newWs[layerIndex].append(newNeuronWs)

                thisEpochError += getEpochError(realOut, _out)
                if isOutputLayer:
                    isOutputLayer = False

            setWs(nw, newWs)

        if lastEpochError is not None:
            epochErrorDiff = abs(lastEpochError - thisEpochError)

            if epochErrorDiff <= trainOptions.minimalEpochErrorDiff:
                training = False
                print(
                    "End {} epoch, last epoch error = {:.2f}, current epoch error = {:.2f} (difference = {:.2f}). Stopping training by reaching minimal epoch error difference - {}".format(
                        epoch, lastEpochError, thisEpochError, epochErrorDiff, trainOptions.minimalEpochErrorDiff))
            else:
                print("End {} epoch, last epoch error = {:.2f}, current epoch error = {:.2f} (difference = {:.2f})".format(epoch,
                                                                                                               lastEpochError,
                                                                                                               thisEpochError,
                                                                                                               epochErrorDiff))
        else:
            print("End {} epoch, epoch error = {:.2f}".format(epoch, thisEpochError))
        lastEpochError = thisEpochError

        if doOnEachEpoch is not None:
            doOnEachEpoch(nw)

        epoch += 1
        if epoch > trainOptions.maxEpoch:
            training = False
            print("Stopping training by reaching maximum epoch - {}".format(trainOptions.maxEpoch))


def getGradient(isOutputLayer: bool, layerIndex: int, neuronIndex: int, nw: NeuralWeb, netsAndOuts: LayersNetsAndOuts,
                gradients: Gradients, realOut: NeuralWebOutput, _out: NeuralWebOutput) -> float:
    neuronActivationFunc = nw.layers[layerIndex][neuronIndex].activationFunc
    neuronNet = netsAndOuts[layerIndex][neuronIndex][0]

    if isOutputLayer:
        error = getError(realOut, _out, neuronIndex)

        gradient = error * neuronActivationFunc.fProizv(neuronNet)
        return gradient
    else:
        sumOfGradWMultiplication = 0
        for nextLayerNeuronIndex, nextLayerNeuron in enumerate(nw.layers[layerIndex + 1]):
            nextLayerNeuronGradient = gradients[layerIndex + 1][nextLayerNeuronIndex]
            nextLayerNeuronWeightToThisNeuron = nextLayerNeuron.weights[neuronIndex]
            sumOfGradWMultiplication += nextLayerNeuronGradient * nextLayerNeuronWeightToThisNeuron

        gradient = neuronActivationFunc.fProizv(neuronNet) * sumOfGradWMultiplication
        return gradient


def getError(realOut: NeuralWebOutput, _out: NeuralWebOutput, neuronIndex: int) -> float:
    return realOut[neuronIndex] - _out[neuronIndex]


def getOut(netsAndOuts: LayersNetsAndOuts, layerIndex: int, wIndex: int, _in: NeuralWebInput) -> float:
    if layerIndex == 0:
        return _in[wIndex]

    return netsAndOuts[layerIndex - 1][wIndex][1]


def setWs(nw: NeuralWeb, newWs: NewWeights):
    for _layerIndex in range(len(newWs), 0, -1):
        layerIndex = _layerIndex - 1
        newWsForThisLayer = newWs[layerIndex]
        for neuronIndex, newWsForThisNeuron in enumerate(newWsForThisLayer):
            nw.layers[layerIndex][neuronIndex].weights = newWsForThisNeuron


def getEpochError(realOut: NeuralWebOutput, _out: NeuralWebOutput):
    error = 0
    for real, expected in zip(realOut, _out):
        diff = real - expected
        error += diff * diff
    return error