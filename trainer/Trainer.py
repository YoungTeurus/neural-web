from typing import Callable

from entities.NeuralWeb import NeuralWeb, NeuralWebInput, NeuralWebOutput, LayersNetsAndOuts

Data = tuple[NeuralWebInput, NeuralWebOutput]
TrainData = list[Data]
Gradients = list[list[float]]
NewWeights = list[list[list[float]]]

DEFAULT_SPEED = 0.5
DEFAULT_MAX_EPOCH = 1000
DEFAULT_ERROR_DECREASE_STOP = 0.05


class TrainOptions:
    speed: float
    maxEpoch: int
    errorDecreaseStop: float

    def __init__(self, speed: float, maxEpoch: int, errorDecreaseStop: float):
        self.speed = speed
        self.maxEpoch = maxEpoch
        self.errorDecreaseStop = errorDecreaseStop

    @classmethod
    def default(cls) -> 'TrainOptions':
        return TrainOptions(speed=DEFAULT_SPEED, maxEpoch=DEFAULT_MAX_EPOCH, errorDecreaseStop=DEFAULT_ERROR_DECREASE_STOP)


def train(nw: NeuralWeb, trainData: TrainData, trainOptions: TrainOptions = None, detailedLogs: bool = False,
          doOnEachEpoch: Callable[[NeuralWeb], None] | None = None) -> None:
    print("Starting train...")

    _trainOptions = trainOptions if trainOptions is not None else TrainOptions.default()

    gradients: Gradients = [[] for _ in range(len(nw.layers))]

    epoch: int = 0
    lastEpochError: float | None = None
    epochErrorDecrease: float | None = None
    epochErrorDecreased: bool = True

    while epoch < _trainOptions.maxEpoch and epochErrorDecreased and (epochErrorDecrease is None or not abs(epochErrorDecrease) < _trainOptions.errorDecreaseStop):
        print("Epoch {} start".format(epoch))
        curEpochError: float = 0
        for td in trainData:
            allNewWeights: NewWeights = []

            _input, expected = td
            actual, netsAndOuts = nw.process(_input)

            epochError = __getEpochError(actual, expected)
            curEpochError += epochError

            outputLayer = True

            for layerIndex in range(len(nw.layers), 0, -1):
                layerIndex = layerIndex - 1
                layer = nw.layers[layerIndex]
                newLayerWs = []

                for (neuronIndex, neuron) in enumerate(layer):
                    net = netsAndOuts[layerIndex][neuronIndex][0]  # net текущего нейрона
                    if outputLayer:
                        gradient = __getError(actual, expected, neuronIndex) * neuron.activationFunc.fProizv(net)
                    else:
                        gradient = neuron.activationFunc.fProizv(net) * __sumOfMultGradients(nw, gradients, layerIndex,
                                                                                             neuronIndex)

                    gradients[layerIndex].append(gradient)

                    newWs = []
                    for (weightIndex, w) in enumerate(neuron.weights):
                        out = __getOut(netsAndOuts, _input, layerIndex, weightIndex)
                        newW = w - _trainOptions.speed * gradient * out
                        newWs.append(newW)
                    newLayerWs.append(newWs)

                allNewWeights.append(newLayerWs)

                if outputLayer:
                    outputLayer = False

            if detailedLogs:
                print("Epoch {} setting weights: {}".format(epoch, allNewWeights))
            __setWs(nw, allNewWeights)

        if doOnEachEpoch is not None:
            doOnEachEpoch(nw)

        epochErrorDecrease = lastEpochError - curEpochError if lastEpochError is not None else None
        epochErrorDecreased = epochErrorDecrease is None or epochErrorDecrease > 0

        if lastEpochError is not None:
            print("Epoch {} error: {:.3f}, last error: {:.3f} (decrease: {:.4f})"
                  .format(epoch, curEpochError, lastEpochError, epochErrorDecrease))
        else:
            print("Epoch {} error: {:.3f}".format(epoch, curEpochError))

        lastEpochError = curEpochError

        epoch += 1

    print("End train, end error: {:.3f}".format(lastEpochError))


def __setWs(nw: NeuralWeb, newWs: NewWeights) -> None:
    '''
    Устанавливает новые веса для нейронов сети
    :param nw: Нейросеть
    :param newWs: Новые веса
    '''
    _newWs = newWs[::-1]
    for layerIndex, layer in enumerate(nw.layers):
        for (neuronIndex, neuron) in enumerate(layer):
            neuron.weights = _newWs[layerIndex][neuronIndex]


def __sumOfMultGradients(nw: NeuralWeb, gradients: Gradients, layerIndex: int, neuronIndex: int) -> float:
    '''
    Высчитывает сумму произведений весов на градиент для каждого нейрона следующего уровня, относительно нейрона
    в слое layerIndex по индексу neuronIndex
    :param nw: Нейросеть
    :param gradients: Градиенты всех нейронов сети, заполненные как минимум до layerIndex + 1 слоя.
    :param layerIndex: Индекс слоя текущего нейрона.
    :param neuronIndex: Индекс текущего нейрона в текущем слое.
    '''
    nextLevelNeurons = nw.layers[layerIndex + 1]
    nextLevelNeuronsGradients = gradients[layerIndex + 1]

    sum = 0
    for (neuron, gradient) in zip(nextLevelNeurons, nextLevelNeuronsGradients):
        weightFromThisToNext = neuron.weights[neuronIndex]
        sum += weightFromThisToNext * gradient

    return sum


def __getOut(netsAndOuts: LayersNetsAndOuts, inputs: NeuralWebInput, layerIndex: int, weightIndex: int) -> float:
    '''
    Возвращает выход нейрона предыдущего слоя для текущего нейрона в слое layerIndex по весу weightIndex.
    :param netsAndOuts: Суммы сумматоров и выходы каждого нейрона сети
    :param inputs: Входы нейросети
    :param layerIndex: Индекс слоя текущего нейрона
    :param weightIndex: Индекс интересующего веса
    '''
    if layerIndex > 0:
        return netsAndOuts[layerIndex - 1][weightIndex][1]
    # Для 1-го слоя, выход предыдущего уровня по данному весу - это вход в нейросеть
    return inputs[weightIndex]


def __getError(actual: NeuralWebOutput, expected: NeuralWebOutput, neuronIndex: int) -> float:
    '''
    Возвращает ошибку нейрона: разницу между ожидаемым значением и имеющимся
    '''
    return actual[neuronIndex] - expected[neuronIndex]


def __getEpochError(actual: NeuralWebOutput, expected: NeuralWebOutput) -> float:
    '''
    Возвращает ошибку нейросети: сумму квадратов разности для каждого нейрона
    '''
    sum = 0
    for pair in zip(actual, expected):
        err = pair[0] - pair[1]
        sum += err * err
    return sum
