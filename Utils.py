import random


def multAndSum(a: list[float], b: list[float]) -> float:
    sum = 0
    for i in zip(a, b):
        sum += i[0] * i[1]
    return sum


def generateWeights(shape: int) -> list[float]:
    weights = []
    for i in range(shape):
        weights.append(random.random())
    return weights


def randomBoundaries(_from: float, _to: float) -> float:
    '''
    Возвращает случайное значение в границах [_from, _to]
    '''
    assert (_from < _to)
    rand = random.random()

    return _from + (_to - _from) * rand


def randomExceptMiddle(_from: float, _fromMax: float, _toMin: float, _to: float) -> float:
    '''
    Возвращает случайное значение в границах [_from, _fromMax) v [_toMin, _to]
    '''
    assert (_from < _fromMax < _toMin < _to)

    rand = random.random()
    if rand >= 0.5:
        return _toMin + (_to - _toMin) * (rand - 0.5) * 2
    else:
        return _from + (_fromMax - _from) * rand * 2
