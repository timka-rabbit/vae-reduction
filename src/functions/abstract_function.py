import numpy as np

from core.data_description import DataDescription


class AbstractFunc(object):
    """
    Интерфейс функции
    """
    def __init__(self, description: DataDescription):
        self._description = description

    @property
    def name(self):
        """
        Название функции
        """
        return self.__class__.__name__

    @property
    def description(self) -> DataDescription:
        """
        Возвращает описание областей определения и значения
        """
        return self._description

    def _verify(self, points: np.ndarray) -> np.ndarray:
        """
        В случае прихода одной точки, оборачиваем её в двумерный массив.
        """
        if points.ndim == 1:
            points = points.reshape(1, points.shape[0])
        assert points.shape[1] >= self._description.x_dim
        if points.shape[1] > self._description.x_dim:
            points = points[:, :self._description.x_dim]
        return points

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Вычисление значения функции в точке
        :param points: ndarray. Двумерный массив точек.
        :returns: ndarray. Двумерный массив значений модели в поданных точках.
        """
        raise NotImplementedError
