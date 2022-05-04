from functions.rosenbrock_func import Rosenbrock

from core.data_handling.generators.regular_grid import Grid
import ui.chart_utils as ui
from unittest import TestCase


class TestRosenbrock(TestCase):
    def test_rosenbrock_2(self):
        func = Rosenbrock()
        x = Grid().get_data(func.description, [100, 100])
        y = func.evaluate(x)
        ui.plot3d(x[:, 0], x[:, 1], y, title=func.name)

    def test_rosenbrock_6(self):
        func = Rosenbrock(n=6)
        x = Grid().get_data(func.description, [5, 5, 5, 5, 5, 5])
        y = func.evaluate(x)
        print(f'X: {x}')
        print(f'Y: {y}')
