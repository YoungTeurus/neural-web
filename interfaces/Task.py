from abc import ABC

from entities.NeuralWeb import NeuralWebInput, NeuralWebOutput


class Task(ABC):
    @classmethod
    def inputShape(cls) -> int:
        '''
        Возвращает размерность входных данных
        '''
        raise NotImplementedError

    @classmethod
    def outputShape(cls) -> int:
        '''
        Возвращает размерность выходных данных
        '''
        raise NotImplementedError

    @classmethod
    def generateInput(cls) -> list[tuple[NeuralWebInput, NeuralWebOutput]]:
        '''
        Возвращает входные данные и ожидаемые выходные данные для нейросети.
        По умолчанию - данный метод подразумевает создание "стандартной задачи", но могут быть и другие методы,
        предполагающие более тонкую настройку.
        '''
        raise NotImplementedError
