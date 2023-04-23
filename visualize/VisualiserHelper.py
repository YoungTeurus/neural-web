from entities.NeuralWeb import NeuralWebInput, NeuralWebOutput, NeuralWeb
from tasks.classification.LinearBinaryClassification import LinearBinaryClassification
from tasks.classification.LinearBinaryClassificationWithTwoOutputNeurons import LinearBinaryClassificationWithTwoOutputNeurons
from tasks.regression.LinearRegression import LinearRegression
from visualize.Visualiser import Visualiser

InAndOutData = list[tuple[NeuralWebInput, NeuralWebOutput]]


class VisualiserHelper:
    @classmethod
    def LinearBinaryClassification(cls, options: 'LinearBinaryClassification.Options',
                                   data: InAndOutData) -> None:
        blues = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 0]
        oranges = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 1]

        lineStart = (0, options.lineB)
        lineEnd = (1, options.lineA * 1 + options.lineB)

        Visualiser.showTwoClassPoints(blues, oranges, [lineStart, lineEnd])

    @classmethod
    def LinearBinaryClassificationResult(cls, options: 'LinearBinaryClassification.Options',
                                         data: InAndOutData, nw: NeuralWeb) -> None:
        blues = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 0]
        oranges = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 1]

        neuron = nw.layers[0][0]
        w1, w2, w3 = neuron.weights
        a = -w1 / w2
        b = -w3 / w2

        lineStart = (0, b)
        lineEnd = (1, a + b)

        optionsLineStart = (0, options.lineB)
        optionsLineEnd = (1, options.lineA * 1 + options.lineB)

        Visualiser.showTwoClassPoints(blues, oranges, [optionsLineStart, optionsLineEnd], [lineStart, lineEnd])

    @classmethod
    def LinearBinaryClassificationWithTwoOutputNeurons(cls,
                                                       options: 'LinearBinaryClassificationWithTwoOutputNeurons.Options',
                                                       data: InAndOutData) -> None:
        blues = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 0]
        oranges = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 1]

        lineStart = (0, options.lineB)
        lineEnd = (1, options.lineA * 1 + options.lineB)

        Visualiser.showTwoClassPoints(blues, oranges, [lineStart, lineEnd])

    @classmethod
    def LinearBinaryClassificationWithTwoOutputNeuronsResult(cls,
                                                             options: 'LinearBinaryClassificationWithTwoOutputNeurons.Options',
                                                             nw: NeuralWeb):
        resultPoints: list[tuple[float, float, float, float]] = []

        numOfPointsX = 50
        numOfPointsY = 50
        xDiff = 1 / numOfPointsX
        yDiff = 1 / numOfPointsY
        for i in range(numOfPointsX):
            x = i * xDiff
            for j in range(numOfPointsY):
                y = j * yDiff

                bProb, oProb = nw.test([[x, y, 1]])[0]
                resultPoints.append((x, y, bProb, oProb))

        lineStart = (0, options.lineB)
        lineEnd = (1, options.lineA * 1 + options.lineB)

        blues = [(point[0], point[1]) for point in resultPoints if point[2] >= point[3]]
        oranges = [(point[0], point[1]) for point in resultPoints if point[2] < point[3]]

        Visualiser.showTwoClassPoints(blues, oranges, [lineStart, lineEnd])

    @classmethod
    def LinearRegression(cls, options: 'LinearRegression.Options',
                         data: InAndOutData) -> None:
        points = [(x[0], y[0]) for x, y in data]

        lineStart = (0, options.lineB)
        lineEnd = (1, options.lineA * 1 + options.lineB)

        Visualiser.showPoints(points, [lineStart, lineEnd])

    @classmethod
    def LinearRegressionResult(cls, nw: NeuralWeb, options: 'LinearRegression.Options', data: InAndOutData) -> None:
        bluePoints = [(x[0], y[0]) for x, y in data]

        amountOfPoints = 50

        xDiff = 1 / amountOfPoints
        xs = [[i * xDiff] for i in range(amountOfPoints)]
        actualXs = [[x[0], 1] for x in xs]
        ys = nw.test(actualXs)

        orangePoints = [(x[0], y[0]) for x, y in zip(xs, ys)]

        lineStart = (0, options.lineB)
        lineEnd = (1, options.lineA * 1 + options.lineB)

        Visualiser.showTwoClassPoints(bluePoints, orangePoints, [lineStart, lineEnd])

    @classmethod
    def XToY(cls, data: InAndOutData):
        points = [(x[0], y[0]) for x, y in data]

        Visualiser.showPoints(points)

    @classmethod
    def SigmoidRegressionResult(cls, nw: NeuralWeb, data: InAndOutData):
        bluePoints = [(x[0], y[0]) for x, y in data]

        amountOfPoints = 50

        xDiff = 1 / amountOfPoints
        xs = [[i * xDiff] for i in range(amountOfPoints)]
        actualXs = [[x[0], 1] for x in xs]
        ys = nw.test(actualXs)

        orangePoints = [(x[0], y[0]) for x, y in zip(xs, ys)]

        Visualiser.showTwoClassPoints(bluePoints, orangePoints)

    @classmethod
    def SigmoidBinaryClassification(cls, data: InAndOutData):
        blues = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 0]
        oranges = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 1]

        Visualiser.showTwoClassPoints(blues, oranges)

    @classmethod
    def SigmoidBinaryClassificationResult(cls, nw: NeuralWeb):
        resultPoints: list[tuple[float, float, float]] = []

        numOfPointsX = 50
        numOfPointsY = 50
        xDiff = 1 / numOfPointsX
        yDiff = 1 / numOfPointsY
        for i in range(numOfPointsX):
            x = i * xDiff
            for j in range(numOfPointsY):
                y = j * yDiff

                _class = nw.test([[x, y, 1]])[0][0]
                resultPoints.append((x, y, _class))

        blues = [(point[0], point[1]) for point in resultPoints if point[2] < 0.5]
        oranges = [(point[0], point[1]) for point in resultPoints if point[2] >= 0.5]

        Visualiser.showTwoClassPoints(blues, oranges)

    @classmethod
    def CircleInsideCircleClassification(cls, data: InAndOutData):
        blues = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 0]
        oranges = [inAndOut[0] for inAndOut in data if inAndOut[1][0] == 1]

        Visualiser.showTwoClassPoints(blues, oranges)

    @classmethod
    def CircleInsideCircleClassificationResult(cls, nw: NeuralWeb):
        resultPoints: list[tuple[float, float, float]] = []

        numOfPointsX = 50
        numOfPointsY = 50
        xDiff = 1 / numOfPointsX
        yDiff = 1 / numOfPointsY
        for i in range(numOfPointsX):
            x = i * xDiff
            for j in range(numOfPointsY):
                y = j * yDiff

                _class = nw.test([[x, y, 1]])[0][0]
                resultPoints.append((x, y, _class))

        blues = [(point[0], point[1]) for point in resultPoints if point[2] < 0.5]
        oranges = [(point[0], point[1]) for point in resultPoints if point[2] >= 0.5]

        Visualiser.showTwoClassPoints(blues, oranges)
