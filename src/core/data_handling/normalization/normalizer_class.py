import numpy as np


class Normalizer(object):
    """
    Класс нормировки данных
    """

    @staticmethod
    def _verify(x: np.ndarray, x_bounds):
        assert x.shape[1] >= len(x_bounds)
        if x.shape[1] > len(x_bounds):
            x_bounds.extend([(0, 1)]*np.abs(x.shape[1] - len(x_bounds)))
        x_dim = len(x_bounds)
        bounds_x_0, bounds_x_1 = zip(*x_bounds)
        delta_x = np.array([bounds_x_1[i] - bounds_x_0[i] for i in range(x_dim)]).reshape(1, x_dim)

        return np.array(bounds_x_0).reshape(1, x_dim), delta_x

    @staticmethod
    def norm(x: np.ndarray, x_bounds) -> np.ndarray:
        """
        Нормировка массива точек
        :param x: np.ndarray. Массив точек для нормировки.
        :param x_bounds: List[Tuple[float, float]]. Границы данных.
        :return: np.ndarray. Нормированный массив точек.
        """
        bounds_0, delta_x = Normalizer._verify(x, x_bounds)

        norm_x = ((x - bounds_0) / delta_x)
        return norm_x

    @staticmethod
    def denorm(x: np.ndarray, x_bounds) -> np.ndarray:
        """
        Денормировка масиива точек
        :param x: np.ndarray. Массив точек для денормировки.
        :param x_bounds: List[Tuple[float, float]]. Границы денормированных данных.
        :return: np.ndarray. Нормированный массив точек.
        """
        bounds_0, delta_x = Normalizer._verify(x, x_bounds)

        x = bounds_0 + (x * delta_x)
        return x
