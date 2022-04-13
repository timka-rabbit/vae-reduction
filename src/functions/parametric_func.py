import numpy as np

from core.data_description import DataDescription
from functions.abstract_function import AbstractFunc


class Ellipse(AbstractFunc):
    """
    Парметрическая кривая в форме эллипса
    """

    def __init__(self, center=(0., 0.), rx=2., ry=1.):
        """
        Конструктор

        :param center: Tuple[float, float]. Центр эллипса.
        :param rx: float. Радиус первой полуоси.
        :param ry: float. Радиос второй полуоси.
        """
        super().__init__(DataDescription(x_dim=1, y_dim=1, x_bounds=[(0, 2*np.pi)]))
        self.x0, self.y0 = center
        self.rx = rx
        self.ry = ry

    @property
    def name(self):
        return "Эллипс"

    def evaluate(self, points):
        x = self.rx * np.cos(points)
        y = self.ry * np.sin(points)
        xy = np.concatenate((x, y), axis=1) + np.array([self.x0, self.y0])
        return xy


class Circle(Ellipse):
    """
    Параметрическая кривая в форме окружности
    """

    def __init__(self, center=(0., 0.), r=1.):
        """
        Конструктор

        :param center: Tuple[float, float]. Центр окружности.
        :param r: float. Радиус окружности.
        """
        super().__init__(center=center, rx=r, ry=r)

    @property
    def name(self):
        return "Окружность"


class Ellipsoid(AbstractFunc):
    """
    Парметрическая поверхность в форме эллипсоида
    """

    def __init__(self, center=(0., 0., 0.), rx=2., ry=1., rz=1.):
        """
        Конструктор

        :param center: Tuple[float, float, float]. Центр эллипсоида.
        :param rx: Радиус первой полуоси.
        :param ry: Радиус второй полуоси.
        :param rz: Радиус третьей полуоси.
        """
        super().__init__(DataDescription(x_dim=1, y_dim=1, x_bounds=[(0, 2*np.pi), (0, np.pi)]))
        self.x0, self.y0, self.z0 = center
        self.rx = rx
        self.ry = ry
        self.rz = rz

    @property
    def name(self):
        return "Эллипсоид"

    def evaluate(self, points):
        u = points[:, 0].reshape((-1, 1))
        v = points[:, 1].reshape((-1, 1))
        x = self.rx * np.cos(u) * np.sin(v)
        y = self.ry * np.sin(u) * np.sin(v)
        z = self.rz * np.cos(v)
        xyz = np.concatenate((x, y, z), axis=1) + np.array([self.x0, self.y0, self.z0])
        return xyz


class Sphere(Ellipsoid):
    """
    Параметрическая поверхность в форме сферы
    """

    def __init__(self, center=(0., 0., 0.), r=1.):
        """
        Конструктор

        :param center: Tuple[float, float, float]. Центр сферы.
        :param r: float. Радиус сферы.
        """
        super().__init__(center=center, rx=r, ry=r, rz=r)

    @property
    def name(self):
        return "Сфера"
