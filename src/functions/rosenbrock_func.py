import numpy as np

from core.data_description import DataDescription
from functions.abstract_function import AbstractFunc


class Rosenbrock(AbstractFunc):
    """
    Функция Розенброка
    """

    def __init__(self):
        super().__init__(DataDescription(x_dim=2, y_dim=1, x_bounds=[(-2, 2), (-2, 2)]))

    @property
    def name(self) -> str:
        return "Функция Розенброка"

    def evaluate(self, points) -> np.ndarray:
        points = self._verify(points)
        return ((1 - points[:, 0]) ** 2 + 100 * (points[:, 1] - points[:, 0] ** 2) ** 2).reshape(-1, 1)
