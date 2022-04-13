import numpy as np

from core.data_description import DataDescription


class AbstractGenerator(object):
    """
    Интерфейс генератора данных
    """

    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        """
        Генерация данных

        :param description: DataDescription. Описание областей определения.
        :param samples_num: int. Количество точек.
        :param irrelevant_var_count: int. Количество незначимых параметров.
        :return: Массив записей.
        """
        raise NotImplementedError

    def _add_irr_vars(self, points: np.ndarray, irrelevant_var_count: int) -> np.ndarray:
        """
        Добавление незначимых переменных

        :param points: ndarray. Массив точек.
        :param irrelevant_var_count: int. Количество незначимых переменных.
        :return: Массив точек, расширенный незначимыми переменными.
        """
        if irrelevant_var_count != 0:
            zeros = [[0] for i in range(irrelevant_var_count)]
            points = np.insert(points, obj=self.dim, values=zeros, axis=1)
        return points
