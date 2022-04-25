import numpy as np
from smt.applications import EGO as EGO_SMT
from core.data_handling.generators.lsh_sequence import LSHSeq
from core.data_description import DataDescription

from core.optimazation.abstract_optimizer import AbstractOptimizer


class EGO(AbstractOptimizer):
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

    def optimize(self, func: callable, n_iter: int) -> np.ndarray:
        """
        Запуск алгоритма оптимизации
        :param func: callable. Функция для оптимизации.
        :param n_iter: int. Количество итераций алгоритма.
        :return: ndarray. Оптимальное решение в виде массива.
        """
        xdoe = LSHSeq().get_data(description=self.data_description,
                                 samples_num=self.data_description.x_dim + 1)
        ego = EGO_SMT(n_iter=n_iter, criterion=self.criterion,
                      xdoe=xdoe, xlimits=np.array(self.data_description.x_bounds))
        x_opt, y_opt, _, _, _ = ego.optimize(fun=func)
        return x_opt.reshape(1, self.data_description.x_dim)
