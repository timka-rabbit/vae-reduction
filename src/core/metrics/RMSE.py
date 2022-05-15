import numpy as np

from core.metrics.abstract_metric import AbstractMetric


class RMSE(AbstractMetric):
    """
    Класс подсчёта корня среднеквадратичной ошибки
    """

    def _do_evaluate(self, actual, expected) -> float:
        """
        :param actual: np.ndarray. Двумерный массив эталонных значений.
        :param expected: np.ndarray. Двумерный массив предсказанных значений.
        :return: float. Скалярное значение ошибки.
        """
        return np.sqrt(np.mean((expected - actual) ** 2))
