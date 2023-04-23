from Utils import randomExceptMiddle, randomBoundaries
from entities.NeuralWeb import NeuralWebInput, NeuralWebOutput
from interfaces.Task import Task


class LinearRegression(Task):
    '''
    Задача линейной регрессии.
    Вход - абсцисса точки
    Выход - ордината точки
    '''

    class Options:
        lineA: float
        lineB: float
        pointCount: int

        def __init__(self, lineA: float, lineB: float, pointCount: int):
            self.lineA = lineA
            self.lineB = lineB
            self.pointCount = pointCount

        @classmethod
        def default(cls) -> 'LinearRegression.Options':
            a = randomExceptMiddle(-1.5, -0.5, 0.5, 1.5)
            b = randomBoundaries(-0.1, 0.1) if a > 0 else randomBoundaries(0.8, 0.9)
            return LinearRegression.Options(a, b, 50)

    @classmethod
    def inputShape(cls) -> int:
        return 2

    @classmethod
    def outputShape(cls) -> int:
        return 1

    @classmethod
    def generateInput(cls) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        return LinearRegression.generateInputOptions()

    @classmethod
    def generateInputOptions(cls, options: Options = None) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        _options = options if options is not None else LinearRegression.Options.default()

        amountOfPoints = options.pointCount
        assert (amountOfPoints > 0)

        inAndOut: list[tuple[NeuralWebInput, NeuralWebOutput]] = []

        xDiff = 1 / amountOfPoints
        for i in range(amountOfPoints):
            x = i * xDiff
            y = options.lineA * x + options.lineB
            y += randomBoundaries(-0.1, 0.1)

            inAndOut.append(([x, 1], [y]))

        return inAndOut
