import math
import random

from Utils import randomExceptMiddle, randomBoundaries
from entities.NeuralWeb import NeuralWebInput, NeuralWebOutput
from interfaces.Task import Task


class SigmoidBinaryClassification(Task):
    '''
    Задача классификации сигмоиды.
    Вход - абсцисса и ордината точки
    Выход - класс точки
    '''

    class Options:
        startY: float
        amplitude: float
        xScale: float
        pointCount: int

        def __init__(self, startY: float, amplitude: float, xScale: float, pointCount: int):
            self.startY = startY
            self.amplitude = amplitude
            self.xScale = xScale
            self.pointCount = pointCount

        @classmethod
        def default(cls) -> 'SigmoidBinaryClassification.Options':
            startY = randomBoundaries(0.4, 0.6)
            amplitude = randomBoundaries(0.2, 0.5)
            xScale = random.randint(5, 50)
            return SigmoidBinaryClassification.Options(startY, amplitude, xScale, 50)

    @classmethod
    def inputShape(cls) -> int:
        return 3

    @classmethod
    def outputShape(cls) -> int:
        return 1

    @classmethod
    def generateInput(cls) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        return SigmoidBinaryClassification.generateInputOptions()

    @classmethod
    def generateInputOptions(cls, options: Options = None) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        _options = options if options is not None else SigmoidBinaryClassification.Options.default()

        amountOfPoints = options.pointCount
        assert (amountOfPoints > 0)

        inAndOut: list[tuple[NeuralWebInput, NeuralWebOutput]] = []

        xDiff = 1 / amountOfPoints
        for i in range(amountOfPoints):
            x = i * xDiff
            y = options.startY + math.sin(x * options.xScale) * options.amplitude
            yRandom = randomExceptMiddle(-0.2, -0.02, 0.02, 0.2)
            y += yRandom

            inAndOut.append(([x, y, 1], [1 if yRandom > 0 else 0]))

        return inAndOut
