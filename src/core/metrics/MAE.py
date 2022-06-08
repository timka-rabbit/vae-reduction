import numpy as np

from core.metrics.abstract_metric import AbstractMetric


class MAE(AbstractMetric):
    """
    Класс подсчёта средней абсолютной ошибки
    """

    def _do_evaluate(self, actual: np.ndarray, expected: np.ndarray) -> float:
        """
        :param actual: np.ndarray. Двумерный массив эталонных значений.
        :param expected: np.ndarray. Двумерный массив предсказанных значений.
        :return: float. Скалярное значение ошибки.
        """
        return np.mean(np.abs(expected - actual))
