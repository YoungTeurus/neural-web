from entities.Neuron import Neuron
from interfaces.ActivationFunc import ActivationFunc

Weights = \
    list[  # For each layer
        list[  # For each neuron
            list[float]]]  # For each weight
Layer = list[Neuron]
NeuralWebShape = list[int]  # [i, l1, l2...] -> i - num of input layers, l1,l2... - num of neurons in each layer
NeuralWebInput = NeuralWebOutput = list[float]

NetAndOut = tuple[float, float]
LayersNetsAndOuts = list[list[NetAndOut]]


class NeuralWeb:
    shape: NeuralWebShape
    layers: list[Layer]

    def __init__(self, shape: NeuralWebShape, layers: list[Layer]):
        self.shape = shape
        self.layers = layers

    @classmethod
    def createNew(cls, shape: NeuralWebShape, activationFuncs: list[ActivationFunc],
                  startingWeights: Weights | None = None) -> 'NeuralWeb':
        '''
        Создаёт новую нейросеть по переданным параметрам.
        :param shape:  Форма нейросети - количество входов, слоёв и нейронов в каждом из них.
        :param activationFuncs: Функция активации для нейронов каждого слоя.
        :param startingWeights: Стартовые веса нейронов (если не задано - случайные)
        '''
        if len(shape) < 2:
            raise RuntimeError("Can't create NeuralWeb without layers")
        if len(activationFuncs) != len(shape) - 1:
            raise RuntimeError("Can't create NeuralWeb with {} layers but {} activationFunctions".format(len(shape) - 1,
                                                                                                         len(activationFuncs)))

        print("Creating NeuralWeb with shape '{}': {} layers, {} inputs and {} outputs".format(shape, len(shape) - 1, shape[0], shape[-1]))

        actualLayers: list[Layer] = []
        for (layerIndex, layer) in enumerate(shape[1:]):
            thisLayer = []
            thisLayerActivationFunc = activationFuncs[layerIndex]
            for i in range(layer):
                thisNeuronWeight = None if startingWeights is None else startingWeights[layerIndex][i]
                thisLayer.append(Neuron.createNew(shape[layerIndex], thisLayerActivationFunc, thisNeuronWeight))
            actualLayers.append(thisLayer)

        return NeuralWeb(shape, actualLayers)

    def process(self, _input: NeuralWebInput) -> tuple[NeuralWebOutput, LayersNetsAndOuts]:
        '''
        Высчитывает выход нейросети и промежуточные результаты (суммы на сумматоре и выходы каждого нейрона)
        :param _input: Входные данные нейросети
        :return: Выход нейросети вместе с промежуточными результатами
        '''
        self.checkInput(_input)

        layersNetsAndOuts: LayersNetsAndOuts = []
        curInput = _input
        neuronIndex = None
        for (layerIndex, layer) in enumerate(self.layers):
            currentLayerNetsAndOuts: list[NetAndOut] = []
            try:
                output = []
                for (neuronIndex, neuron) in enumerate(layer):
                    net = neuron.getNet(curInput)
                    out = neuron.activationFunc.f(net)
                    output.append(out)
                    currentLayerNetsAndOuts.append((net, out))
                curInput = output
                layersNetsAndOuts.append(currentLayerNetsAndOuts)
            except RuntimeError as e:
                nIndex = neuronIndex if neuronIndex is not None else "N/A"
                raise RuntimeError(
                    "There was an exception on the {} layer at {} neuron: {}".format(layerIndex, nIndex, e))

        return curInput, layersNetsAndOuts

    def test(self, _inputs: list[NeuralWebInput]) -> list[NeuralWebOutput]:
        '''
        Высчитывает выход нейросети
        :param _input: Входные данные нейросети
        :return: Выход нейросети
        '''
        return [self.process(_input)[0] for _input in _inputs]

    def __str__(self) -> str:
        out = ""
        for (li, layer) in enumerate(self.layers):
            if li > 0:
                out += "\n"
            strLayer = [str(n) for n in layer]
            out += "Layer {}:".format(li) + ",".join(strLayer)
        return out

    def checkInput(self, _input: NeuralWebInput) -> None:
        expectedShape = self.shape[0]
        inputShape = len(_input)
        if inputShape != expectedShape:
            raise RuntimeError("Invalid shape of input: expected {}, but was {}".format(expectedShape, inputShape))
