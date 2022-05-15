import numpy as np
from core.data_description import DataDescription
from core.data_handling.generators.regular_grid import Grid

from core.optimazation.abstract_optimizer import AbstractOptimizer


class BruteForce(AbstractOptimizer):
    def __init__(self, data_description: DataDescription, grid_count: int = 10):
        """
        :param data_description: DataDescription. Описание области определения.
        :param gridDivCount: int. Параметр разбиения сетки.
        """
        super().__init__(data_description=data_description)
        self.grid_count = grid_count

    def optimize(self, func: callable, n_iter: int = None, is_min: bool = True) -> np.ndarray:
        """
        Запуск алгоритма оптимизации
        :param func: callable. Функция для оптимизации.
        :param n_iter: int. Количество итераций алгоритма.
        :param is_min: bool. True - поиск минимума, False - поиск максимума.
        :return: ndarray. Оптимальное решение в виде массива.
        """

        extr = float('inf') if is_min is True else float('-inf')
        if is_min is False:
            fun = lambda x: 1 / func(x)
        else:
            fun = func

        permuts = Grid().get_data(description=self.data_description,
                                  samples_num=[self.grid_count for i in range(self.data_description.x_dim)])

        res = None
        for x in permuts:
            val = fun(x)
            if (is_min and val < extr) or (not is_min and val > extr):
                extr = val
                res = x.copy()
        return np.array(res)
