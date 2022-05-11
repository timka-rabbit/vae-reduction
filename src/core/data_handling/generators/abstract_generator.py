import numpy as np
import random
import time
from typing import Union

from core.data_description import DataDescription


class AbstractGenerator(object):
    """
    Интерфейс генератора данных
    """

    def get_data(self, description: DataDescription, samples_num: Union[int, list],
                 irrelevant_var_count: int = 0) -> np.ndarray:
        """
        Генерация данных
        :param description: DataDescription. Описание областей определения и значений.
        :param samples_num: Union[int, list]. Для генераторов указывается количество точек.
        Для сетки нужно указывать явно число точек для каждой координаты в виде списка.
        :param irrelevant_var_count: int. Количество незначимых параметров.
        :return: Массив записей.
        """
        raise NotImplementedError

    def _add_irr_vars(self,  description: DataDescription,
                      points: np.ndarray, irrelevant_var_count: int) -> np.ndarray:
        """
        Добавление незначимых переменных
        :param description: DataDescription. Описание облпстей определения и значений.
        :param points: ndarray. Массив точек.
        :param irrelevant_var_count: int. Количество незначимых переменных.
        :return: Массив точек, расширенный незначимыми переменными.
        """
        random.seed(a=time.time())
        if irrelevant_var_count != 0:
            zeros = np.zeros((points.shape[0], irrelevant_var_count))
            for i in range(points.shape[0]):
                for j in range(irrelevant_var_count):
                    zeros[i, j] = random.uniform(0, 1)
            points = np.hstack((points, zeros))
        return points
