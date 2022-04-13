from typing import Tuple, List


class DataDescription(object):
    """
    Класс описания областей опредления и значения
    """
    def __init__(self, x_dim, y_dim, x_bounds=None, y_bounds=None):
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
