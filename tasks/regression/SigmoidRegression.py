import math
import random

from Utils import randomExceptMiddle, randomBoundaries
from entities.NeuralWeb import NeuralWebInput, NeuralWebOutput
from interfaces.Task import Task


class SigmoidRegression(Task):
    '''
    Задача регрессии сигмоиды.
    Вход - абсцисса точки
    Выход - ордината точки
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
        def default(cls) -> 'SigmoidRegression.Options':
            startY = randomBoundaries(0.4, 0.6)
            amplitude = randomBoundaries(0.2, 0.5)
            xScale = random.randint(5, 50)
            return SigmoidRegression.Options(startY, amplitude, xScale, 50)

    @classmethod
    def inputShape(cls) -> int:
        return 2

    @classmethod
    def outputShape(cls) -> int:
        return 1

    @classmethod
    def generateInput(cls) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        return SigmoidRegression.generateInputOptions()

    @classmethod
    def generateInputOptions(cls, options: Options = None) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        _options = options if options is not None else SigmoidRegression.Options.default()

        amountOfPoints = options.pointCount
        assert (amountOfPoints > 0)

        inAndOut: list[tuple[NeuralWebInput, NeuralWebOutput]] = []

        xDiff = 1 / amountOfPoints
        for i in range(amountOfPoints):
            x = i * xDiff
            y = options.startY + math.sin(x * options.xScale) * options.amplitude
            y += randomBoundaries(-0.05, 0.05)

            inAndOut.append(([x, 1], [y]))

        return inAndOut
