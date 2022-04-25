import numpy as np

from core.data_description import DataDescription
from functions.abstract_function import AbstractFunc


class Rastrigin(AbstractFunc):
    """
    Функция Растригина размерности n
    """

    def __init__(self, n: int, a: float = 10):
        """
        :param n: int. Размерность области определения.
        :param a: float. Нормирующий параметр.
        """
        super().__init__(DataDescription(x_dim=n, y_dim=1, x_bounds=[(-5.12, 5.12)]*n))
        self.n = n
        self.a = a

    @property
    def name(self) -> str:
        return f'Функция Растригина размерности {self.description.x_dim}'

    def evaluate(self, points):
        points = self._verify(points)
        return (self.a * self.n + np.sum(points ** 2 - self.a * np.cos(2 * np.pi * points), axis=1)).reshape(-1, 1)
