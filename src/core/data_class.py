import numpy as np
from core.data_handling.normalization.normalizer_class import Normalizer
from core.data_description import DataDescription


class Data(object):
    def __init__(self, x, y, description=None):
        """
        Конструктор

        :param x: ndarray. Двумерный массив точек из области определения.
        :param y: ndarray. Двумерный массив точек из области значений.
        :param description: DataDescription. Описание областей определения и значений (опционально)
        """
        self.x = x
        self.y = y
        self._description = description
        self.init()

    def init(self):
        """
        Проверка и инициализация свойств объекта
        """
        assert self.x.ndim == 2
        assert self.y.ndim == 2
        assert self.x.shape[0] == self.y.shape[0]
        if self._description is None:
            x_dim = self.x.shape[1]
            y_dim = self.y.shape[1]
            x_bounds = DataDescription.calculate_bounds(self.x)
            y_bounds = DataDescription.calculate_bounds(self.y)
            self._description = DataDescription(x_dim=x_dim, y_dim=y_dim, x_bounds=x_bounds, y_bounds=y_bounds)
        else:
            assert self._description.x_dim == self.x.shape[1]
            assert self._description.y_dim == self.y.shape[1]
            if self._description.x_bounds is None:
                x_bounds = DataDescription.calculate_bounds(self.x)
            else:
                x_bounds = self._description.x_bounds
            if self._description.y_bounds is None:
                y_bounds = DataDescription.calculate_bounds(self.y)
            else:
                y_bounds = self._description.y_bounds
            self._description = DataDescription(x_dim=self._description.x_dim, y_dim=self._description.y_dim,
                                                x_bounds=x_bounds, y_bounds=y_bounds)

    def add(self, x, y):
        """
        Добавление новой точки и значения в ней

        :param x: Новая точка.
        :param y: Значение функции в точке.
        """
        self.x = np.vstack(self.x, x)
        self.y = np.vstack(self.y, y)

    @property
    def count(self) -> int:
        """
        Число объектов данных
        """
        return self.x.shape[0]

    @property
    def description(self) -> DataDescription:
        """
        Описание данных
        """
        return self._description

    def get_x_norm(self):
        """
        Получение нормированных значений x.
        """
        return Normalizer.norm(self.x, self._description.x_bounds)

    def get_y_norm(self):
        """
        Получение нормированных значений y.
        """
        return Normalizer.norm(self.y, self._description.y_bounds)

    def get_x_denorm(self, x_bounds):
        """
        Получение денормированных значений x.
        :param x_bounds: List[Tuple(float, float)]. Границы x для денормировки.
        """
        assert len(self._description.x_bounds) == len(x_bounds)
        return Normalizer.denorm(self.x, x_bounds)

    def get_y_denorm(self, y_bounds):
        """
        Получение денормированных значений y.
        :param y_bounds: List[Tuple(float, float)]. Границы y для денормировки.
        """
        assert len(self._description.y_bounds) == len(y_bounds)
        return Normalizer.denorm(self.y, y_bounds)
