import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription


class GridSeq(AbstractGenerator):
    """
    Генерация данных на равномерной сетке
    """
    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        axes_num = int(samples_num**(1/2))
        x_limits = description.x_bounds
        dimensions = len(x_limits)
        # создание значений по каждой координате
        dim_values = [np.linspace(x_limits[dim][0], x_limits[dim][1], axes_num) for dim in range(dimensions)]
        # дублирование значений по координатам для создания сетки
        meshed_values = np.meshgrid(*dim_values)
        meshed_values = [v.ravel() for v in meshed_values]
        # объединение значений в точки размерности (m, n) где m число точек в сетке, n - размерность пространства
        points = np.stack(meshed_values).T
        print(points)
        points = self._add_irr_vars(points, irrelevant_var_count)
        return points
