import numpy as np

from core.data_handling.normalization.normalizer_class import Normalizer

from functions.parametric_func import Ellipse
from core.data_handling.generators.grid_sequence import GridSeq as Grid
from core.data_class import Data
import ui.chart_utils as cu
from unittest import TestCase


class TestNormalization(TestCase):
    def test_ellipse_norm_1(self):
        func = Ellipse(rx=4, ry=2)
        x = Grid().get_data(func.description, 100)
        y = func.evaluate(x)
        data = Data(y[:, 0, np.newaxis], y[:, 1, np.newaxis])
        cu.plot2d(data.x, data.y, 'Обычный график')
        norm_data = Normalizer.norm(data)
        cu.plot2d(norm_data.x, norm_data.y, 'Нормированный график')

    def test_ellipse_norm_2(self):
        func = Ellipse(rx=4, ry=2)
        x = Grid().get_data(func.description, 100)
        y = func.evaluate(x)
        data = Data(y[:, 0, np.newaxis], y[:, 1, np.newaxis])
        cu.plot2d(data.x, data.y, 'Обычный график')
        norm_x = Normalizer.norm(data.x, x_bounds=[(-4, 4)])
        norm_y = Normalizer.norm(data.y, x_bounds=[(-2, 2)])
        cu.plot2d(norm_x, norm_y, 'Нормированный график')

    def test_ellipse_denorm_1(self):
        func = Ellipse(center=(0.5, 0.5), rx=0.5, ry=0.5)
        x = Grid().get_data(func.description, 100)
        y = func.evaluate(x)
        data = Data(y[:, 0, np.newaxis], y[:, 1, np.newaxis])
        cu.plot2d(data.x, data.y, 'Обычный график')
        denorm_data = Normalizer.denorm(data, x_bounds=[(-4, 4)], y_bounds=[(-2, 2)])
        cu.plot2d(denorm_data.x, denorm_data.y, 'Денормированный график')

    def test_ellipse_denorm_2(self):
        func = Ellipse(center=(0.5, 0.5), rx=0.5, ry=0.5)
        x = Grid().get_data(func.description, 100)
        y = func.evaluate(x)
        data = Data(y[:, 0, np.newaxis], y[:, 1, np.newaxis])
        cu.plot2d(data.x, data.y, 'Обычный график')
        denorm_x = Normalizer.denorm(data.x, x_bounds=[(-4, 4)])
        denorm_y = Normalizer.denorm(data.y, x_bounds=[(-2, 2)])
        cu.plot2d(denorm_x, denorm_y, 'Денормированный график')
