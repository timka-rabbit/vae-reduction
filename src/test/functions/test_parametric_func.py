from functions.parametric_func import Ellipse
from functions.parametric_func import Circle
from functions.parametric_func import Ellipsoid
from functions.parametric_func import Sphere

from core.data_handling.generators.regular_grid import Grid
import ui.chart_utils as ui
from unittest import TestCase


class TestParametricFunc(TestCase):
    def test_ellipse(self):
        func = Ellipse()
        x = Grid().get_data(func.description, [100, 100])
        y = func.evaluate(x)
        ui.plot2d(y[:, 0], y[:, 1], title=func.name)

    def test_circle(self):
        func = Circle()
        x = Grid().get_data(func.description, [100, 100])
        y = func.evaluate(x)
        ui.plot2d(y[:, 0], y[:, 1], title=func.name)

    def test_ellipsoid(self):
        func = Ellipsoid()
        x = Grid().get_data(func.description, [100, 100])
        z = func.evaluate(x)
        ui.plot3d(z[:, 0], z[:, 1], z[:, 2], title=func.name)

    def test_sphere(self):
        func = Sphere()
        x = Grid().get_data(func.description, [100, 100])
        z = func.evaluate(x)
        ui.plot3d(z[:, 0], z[:, 1], z[:, 2], title=func.name)
