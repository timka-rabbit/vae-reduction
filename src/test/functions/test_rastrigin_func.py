from functions.rastrigin_func import Rastrigin

from core.data_handling.generators.regular_grid import Grid
import ui.chart_utils as ui
from unittest import TestCase


class TestRastrigin(TestCase):
    def test_rastrigin_2(self):
        func = Rastrigin(n=2)
        x = Grid().get_data(func.description, [100, 100])
        y = func.evaluate(x)
        ui.plot3d(x[:, 0], x[:, 1], y, title=func.name)

    def test_rastrigin_6(self):
        func = Rastrigin(n=6)
        x = Grid().get_data(func.description, [5, 5, 5, 5, 5, 5])
        y = func.evaluate(x)
        print(f'X: {x}')
        print(f'Y: {y}')
