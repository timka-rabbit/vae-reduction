import numpy as np
from scipy.optimize import minimize
from core.data_handling.generators.lhs_sequence import LHS
from core.data_description import DataDescription

from core.optimazation.abstract_optimizer import AbstractOptimizer


class Minimizer(AbstractOptimizer):
    def __init__(self, data_description: DataDescription, method: str = 'BFGS'):
        """
        :param data_description: DataDescription. Описание области определения.
        :param method: str. Используемый метод оптимизации:
            'Nelder-Mead' - метод Нельдера-Мида
            'Powell' - метод сопряжённых направлений
            'CG' - метод сопряжённых градиентов
            'BFGS' - алгоритм Бройдена-Флетчера-Гольдфарба-Шанно
            'COBYLA' - ограниченная оптимизация с помощью линейной аппроксимации
            'SLSQP' - последовательное программирование методом наименьших квадратов.
        """
        super().__init__(data_description=data_description)
        self.method = method

    def optimize(self, func: callable, n_iter: int) -> np.ndarray:
        """
        Запуск алгоритма оптимизации
        :param func: callable. Функция для оптимизации.
        :param n_iter: int. Количество итераций алгоритма.
        :return: ndarray. Оптимальное решение в виде массива.
        """
        x0 = LHS().get_data(description=self.data_description,
                            samples_num=self.data_description.x_dim + 1)[0].reshape(1, self.data_description.x_dim)
        res = minimize(fun=func, x0=x0, method=self.method,
                       options={'maxiter': n_iter})
        return res.x
