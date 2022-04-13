import numpy as np
from typing import Tuple, List


class DataDescription(object):
    """
    Класс описания областей опредления и значения
    """
    def __init__(self, x_dim, y_dim, x_bounds=None, y_bounds=None):
        """
        Конструктор

        :param x_dim: int. Размерность области определения.
        :param y_dim: int. Размерность области значений.
        :param x_bounds: List[Tuple[float, float]]. Границы данных области определения.
        :param y_bounds: List[Tuple[float, float]]. Границы данных области значений.
        """
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds

    @property
    def x_dim(self) -> int:
        """
        Размерность области определения
        """
        return self._x_dim

    @property
    def y_dim(self) -> int:
        """
        Размерность области значений
        """
        return self._y_dim

    @property
    def x_bounds(self) -> List[Tuple[float, float]]:
        """
        Границы области определения
        """
        return self._x_bounds

    @property
    def y_bounds(self) -> List[Tuple[float, float]]:
        """
        Границы области значений
        """
        return self._y_bounds

    @staticmethod
    def calculate_bounds(data) -> List[Tuple[float, float]]:
        """
        Вычисление граничных значений в наборе данных

        :param data: ndarray. Двумерный массив точек.
        :returns: List[Tuple[float, float]]. Список границ.
        """
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        limits = np.vstack([data_min, data_max])
        return [(limits[0][i], limits[1][i]) for i in range(limits.shape[1])]
