import numpy as np


class AbstractMetric(object):
    """
    Интерфейс метрик
    """

    def evaluate(self, actual, expected) -> float:
        """
        :param actual: np.ndarray. Двумерный массив эталонных значений.
        :param expected: np.ndarray. Двумерный массив предсказанных значений.
        :return: float. Скалярное значение метрики.
        """
        assert isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
        assert actual.ndim == 2
        assert expected.shape == actual.shape
        return self._do_evaluate(actual, expected)

    def _do_evaluate(self, actual: np.ndarray, expected: np.ndarray):
        """
        Внутренний метод подсчета значения метрики
        """
        raise NotImplementedError
