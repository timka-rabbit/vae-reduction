import numpy as np

from core.data_description import DataDescription
from functions.abstract_function import AbstractFunc


class Rosenbrock(AbstractFunc):
    """
    Функция Розенброка размерности n
    """

    def __init__(self, n=2):
        """
        :param n: int. Размерность функции.
        """
        super().__init__(DataDescription(x_dim=n, y_dim=1, x_bounds=[(-2, 2)]*n))
        self.n = n

    @property
    def name(self) -> str:
        return f'Функция Розенброка размерности {self.description.x_dim}'

    def evaluate(self, points) -> np.ndarray:
        points = self._verify(points)
        y = np.zeros((points.shape[0],))
        for j in range(points.shape[0]):
            for i in range(self.n - 1):
                y[j] += (1 - points[j, i]) ** 2 + 100 * (points[j, i + 1] - points[j, i] ** 2) ** 2
        return y.reshape(-1, 1)
