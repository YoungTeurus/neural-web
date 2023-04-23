import matplotlib.pyplot as plt

DrawPoint= tuple[float, float] | list[float]  # x, y
ColoredDrawPoint = tuple[float, float, str]  # x, y, color


class Visualiser:
    @classmethod
    def showPoints(cls, points: list[DrawPoint], line: list[DrawPoint] = None) -> None:
        pointsWithColor = [(point[0], point[1], 'blue') for point in points]

        Visualiser.__show(pointsWithColor, line)

    @classmethod
    def showTwoClassPoints(cls, bluePoints: list[DrawPoint], orangePoints: list[DrawPoint], line: list[DrawPoint] = None, otherLine: list[DrawPoint] = None) -> None:
        bluePointsWithColor = [(point[0], point[1], 'blue') for point in bluePoints]
        orangePointsWithColor = [(point[0], point[1], 'orange') for point in orangePoints]

        Visualiser.__show(bluePointsWithColor + orangePointsWithColor, line, otherLine)

    @classmethod
    def __show(cls, points: list[ColoredDrawPoint], line: list[DrawPoint] = None, otherLine: list[DrawPoint] = None) -> None:
        x: list[float] = [point[0] for point in points]
        y: list[float] = [point[1] for point in points]
        colors: list[str] = [point[2] for point in points]

        plt.scatter(x, y, color=colors)
        if line is not None:
            lineX: list[float] = [point[0] for point in line]
            lineY: list[float] = [point[1] for point in line]
            plt.plot(lineX, lineY, color='grey')

        if otherLine is not None:
            lineX: list[float] = [point[0] for point in otherLine]
            lineY: list[float] = [point[1] for point in otherLine]
            plt.plot(lineX, lineY, color='red')

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.xlim(0, 1)
        plt.ylim(0, 1)

        plt.show()
