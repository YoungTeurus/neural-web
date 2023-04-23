import random

from Utils import randomBoundaries
from entities.NeuralWeb import NeuralWebInput, NeuralWebOutput
from interfaces.Task import Task


class CircleInsideCircleClassification(Task):
    '''
    Задача классификации круга внутри круга.
    Вход - абсцисса и ордината точки
    Выход - класс точки
    '''

    class Options:
        circlesX: float
        circlesY: float
        innerCircleRadius: float
        pointCount: int

        def __init__(self, circlesX: float, circlesY: float, innerCircleRadius: float, pointCount: int):
            self.circlesX = circlesX
            self.circlesY = circlesY
            self.innerCircleRadius = innerCircleRadius
            self.pointCount = pointCount

        @classmethod
        def default(cls) -> 'CircleInsideCircleClassification.Options':
            circlesX = 0.5
            circlesY = 0.5
            innerCircleRadius = randomBoundaries(0.05, 0.15)
            return CircleInsideCircleClassification.Options(circlesX, circlesY, innerCircleRadius, 150)

    @classmethod
    def inputShape(cls) -> int:
        return 3

    @classmethod
    def outputShape(cls) -> int:
        return 1

    @classmethod
    def generateInput(cls) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        return CircleInsideCircleClassification.generateInputOptions()

    @classmethod
    def generateInputOptions(cls, options: Options = None) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        _options = options if options is not None else CircleInsideCircleClassification.Options.default()

        amountOfPoints = options.pointCount
        assert (amountOfPoints > 0)

        inAndOut: list[tuple[NeuralWebInput, NeuralWebOutput]] = []

        for i in range(amountOfPoints):
            x = random.random()
            y = random.random()

            _x = x - options.circlesX
            _y = y - options.circlesY

            _class = 1 if _x * _x + _y * _y <= options.innerCircleRadius else 0

            inAndOut.append(([x, y, 1], [_class]))

        return inAndOut
