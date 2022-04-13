import numpy as np
from typing import Tuple, List

from core.data_description import DataDescription
from core.data_class import Data


class Normalizer(object):
    """
    Класс нормировки данных
    """

    @staticmethod
    def norm(data, norm_min: float = 0., norm_max: float = 1.) -> Data:
        """
        Нормировка данных

        :param data: Data. Данные для нормировки.
        :param norm_min: float. Нижняя граница нормированных данных (по умолчанию 0).
        :param norm_max: float. Верхняя граница нормированных данных (по умолчанию 1).
        :return: Data. Нормированные данные.
        """
        x_dim = data.description.x_dim
        y_dim = data.description.y_dim
        bounds_x_0, bounds_x_1 = zip(*data.description.x_bounds)
        delta_x = np.array([bounds_x_0[i] - bounds_x_1[i] for i in range(x_dim)])
        bounds_y_0, bounds_y_1 = zip(*data.description.y_bounds)
        delta_y = np.array([bounds_y_0[i] - bounds_y_1[i] for i in range(y_dim)])
        delta_norm = norm_max - norm_min

        norm_x = norm_min + ((data.x.copy() - np.array(bounds_x_0)) * delta_norm / delta_x)
        norm_y = norm_min + ((data.y.copy() - np.array(bounds_y_0)) * delta_norm / delta_y)
        return Data(x=norm_x, y=norm_y,
                    description=DataDescription(x_dim=data.description.x_dim,
                                                y_dim=data.description.y_dim,
                                                x_bounds=[(0., 1.) for i in range(x_dim)],
                                                y_bounds=[(0., 1.) for i in range(y_dim)]))

    @staticmethod
    def denorm(data, x_bounds, y_bounds) -> Data:
        """
        Денормировка данных

        :param data: Data. Нормированные данные для денормировки.
        :param x_bounds: List[Tuple[float, float]]. Границы данных x для денормирования.
        :param y_bounds: List[Tuple[float, float]]. Границы данных y для денормирования.
        :return: Data. Денормированные данные.
        """
        x_dim = data.description.x_dim
        y_dim = data.description.y_dim
        bounds_x_0_norm, bounds_x_1_norm = zip(*data.description.x_bounds)
        delta_x_norm = np.array([bounds_x_0_norm[i] - bounds_x_1_norm[i] for i in range(x_dim)])
        bounds_y_0_norm, bounds_y_1_norm = zip(*data.description.y_bounds)
        delta_y_norm = np.array([bounds_y_0_norm[i] - bounds_y_1_norm[i] for i in range(y_dim)])

        bounds_x_0, bounds_x_1 = zip(*x_bounds)
        delta_x = np.array([bounds_x_0[i] - bounds_x_1[i] for i in range(x_dim)])
        bounds_y_0, bounds_y_1 = zip(*y_bounds)
        delta_y = np.array([bounds_y_0[i] - bounds_y_1[i] for i in range(y_dim)])

        x = np.array(bounds_x_0) + ((data.x.copy() - np.array(bounds_x_0_norm)) / delta_x_norm * delta_x)
        y = np.array(bounds_y_0) + ((data.y.copy() - np.array(bounds_y_0_norm)) / delta_y_norm * delta_y)
        return Data(x=x, y=y,
                    description=DataDescription(x_dim=data.description.x_dim,
                                                y_dim=data.description.y_dim,
                                                x_bounds=x_bounds,
                                                y_bounds=y_bounds))
