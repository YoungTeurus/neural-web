import random

from Utils import randomExceptMiddle, randomBoundaries
from entities.NeuralWeb import NeuralWebInput, NeuralWebOutput
from interfaces.Task import Task


class LinearBinaryClassificationWithTwoOutputNeurons(Task):
    '''
    Задача линейного бинарного разделения.
    Входные данные: x и y точек в диапазоне [0,1].
    Выходные данные: номер класса - целое число 0 или 1.
    '''

    class Options:
        lineA: float
        lineB: float
        amountOfPoints: int

        def __init__(self, lineA: float, lineB: float, amountOfPoints: int):
            self.lineA = lineA
            self.lineB = lineB
            self.amountOfPoints = amountOfPoints

        @classmethod
        def default(cls) -> 'LinearBinaryClassificationWithTwoOutputNeurons.Options':
            a = randomExceptMiddle(-1.5, -0.5, 0.5, 1.5)
            b = randomBoundaries(-0.1, 0.1) if a > 0 else randomBoundaries(0.8, 0.9)
            return LinearBinaryClassificationWithTwoOutputNeurons.Options(a, b, 50)


    @classmethod
    def inputShape(cls) -> int:
        return 3

    @classmethod
    def outputShape(cls) -> int:
        return 2

    @classmethod
    def generateInput(cls) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        return LinearBinaryClassificationWithTwoOutputNeurons.generateInputOptions()

    @classmethod
    def generateInputOptions(cls, options: Options = None) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        _options = options if options is not None else LinearBinaryClassificationWithTwoOutputNeurons.Options.default()

        amountOfPoints = options.amountOfPoints
        assert(amountOfPoints > 0)

        inAndOut: list[tuple[NeuralWebInput, NeuralWebOutput]] = []

        xDiff = 1 / amountOfPoints
        for i in range(amountOfPoints):
            x = 0 + xDiff * i
            y = random.random()
            point = [x, y, 1]
            _class = LinearBinaryClassificationWithTwoOutputNeurons.getPointClass(options, point)

            inAndOut.append((point, [_class, 0 if _class == 1 else 1]))

        return inAndOut

    @classmethod
    def getPointClass(cls, options: 'LinearBinaryClassificationWithTwoOutputNeurons.Options', point: list[float]) -> int:
        # Ниже линии - 0, выше линии - 1
        if options.lineA * point[0] + options.lineB < point[1]:
            return 1
        return 0
