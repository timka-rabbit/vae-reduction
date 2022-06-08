import numpy as np
from core.data_description import DataDescription


class AbstractOptimizer(object):
    """
    Абстрактный класс алгоритмов оптимизации
    """

    def __init__(self, data_description: DataDescription):
        """
        :param data_description: DataDescription. Описание областей определения и значений.
        """
        self.data_description = data_description

    def optimize(self, func: callable, n_iter: int, is_min: bool = True) -> np.ndarray:
        """
        Запуск алгоритма оптимизации
        :param func: callable. Функция для оптимизации.
        :param n_iter: int. Количество итераций алгоритма.
        :param is_min: bool. True - поиск минимума, False - поиск максимума.
        :return: ndarray. Оптимальное решение в виде массива.
        """
        raise NotImplementedError
