import numpy as np

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
        self.description = description
        self.init()

    def init(self):
        """
        Проверка и инициализация свойств объекта
        """
        assert self.x.ndim == 2
        assert self.y.ndim == 2
        assert self.x.shape[0] == self.y.shape[0]
        if self.description is None:
            x_dim = self.x.shape[1]
            y_dim = self.y.shape[1]
            x_bounds = DataDescription.calculate_bounds(self.x)
            y_bounds = DataDescription.calculate_bounds(self.y)
            self.description = DataDescription(x_dim=x_dim, y_dim=y_dim, x_bounds=x_bounds, y_bounds=y_bounds)
        else:
            assert self.description.x_dim == self.x.shape[1]
            assert self.description.y_dim == self.y.shape[1]

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
        return self.description
