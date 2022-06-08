import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription


class Grid(AbstractGenerator):
    """
    Генерация данных на равномерной сетке
    """

    def get_data(self, description: DataDescription, samples_num: list,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        """
        Генерация данных
        :param description: DataDescription. Описание областей определения и значений.
        :param samples_num: Union[int, list]. Для генераторов указывается количество точек.
        Для сетки нужно указывать явно число точек для каждой координаты в виде списка.
        :param irrelevant_var_count: int. Количество незначимых параметров.
        :return: Массив записей.
        """
        x_limits = description.x_bounds
        dimensions = description.x_dim
        # создание значений по каждой координате
        dim_values = [np.linspace(x_limits[dim][0], x_limits[dim][1], samples_num[dim]) for dim in range(dimensions)]
        # дублирование значений по координатам для создания сетки
        meshed_values = np.meshgrid(*dim_values)
        meshed_values = [v.ravel() for v in meshed_values]
        # объединение значений в точки размерности (m, n) где m число точек в сетке, n - размерность пространства
        points = np.stack(meshed_values).T
        points = self._add_irr_vars(points, irrelevant_var_count)
        return points
