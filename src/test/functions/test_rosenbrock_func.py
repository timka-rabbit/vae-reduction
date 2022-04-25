from functions.rosenbrock_func import Rosenbrock

from core.data_handling.generators.regular_grid import Grid
import ui.chart_utils as ui
from unittest import TestCase


class TestRosenbrock(TestCase):
    def test_rosenbrock(self):
        func = Rosenbrock()
        x = Grid().get_data(func.description, 10000)
        y = func.evaluate(x)
        ui.plot3d(x[:, 0], x[:, 1], y, title=func.name)
