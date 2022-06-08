import numpy as np
from smt.applications import EGO as EGO_SMT
from core.data_handling.generators.lhs_sequence import LHS
from core.data_description import DataDescription

from core.optimazation.abstract_optimizer import AbstractOptimizer


class EGO(AbstractOptimizer):
    """
    Алгоритм эффективной глобальной оптимизации
    """

    def __init__(self, data_description: DataDescription, criterion: str = 'EI'):
        """
        :param data_description: DataDescription. Описание области определения.
        :param criterion: str. Критерий выбора следующей точки:
            'EI' - Ожидаемое улучшение
            'SBO' - Cуррогатная оптимизация
            'LCB' - Нижняя доверительная граница.
        """
        super().__init__(data_description=data_description)
        self.criterion = criterion

    def optimize(self, func: callable, n_iter: int, is_min: bool = True) -> np.ndarray:
        """
        Запуск алгоритма оптимизации
        :param func: callable. Функция для оптимизации.
        :param n_iter: int. Количество итераций алгоритма.
        :param is_min: bool. True - поиск минимума, False - поиск максимума.
        :return: ndarray. Оптимальное решение в виде массива.
        """
        xdoe = LHS().get_data(description=self.data_description,
                              samples_num=self.data_description.x_dim + 1)
        ego = EGO_SMT(n_iter=n_iter, criterion=self.criterion,
                      xdoe=xdoe, xlimits=np.array(self.data_description.x_bounds))
        if is_min is False:
            fun = lambda x: 1 / func(x)
        else:
            fun = func
        x_opt, y_opt, _, _, _ = ego.optimize(fun=fun)
        return x_opt.reshape(1, self.data_description.x_dim)
