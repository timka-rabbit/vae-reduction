import numpy as np

from core.metrics.MAE import MAE
from core.metrics.RMSE import RMSE

from functions.parametric_func import Ellipse
from core.data_handling.generators.regular_grid import Grid
from unittest import TestCase


class TestMetrics(TestCase):
    def test_ellipse(self):
        el_1 = Ellipse(center=(0, 0), rx=2, ry=1)
        data_1 = el_1.evaluate(Grid().get_data(el_1.description, [100, 100]))
        x_1 = data_1[:, 0, np.newaxis]
        y_1 = data_1[:, 1, np.newaxis]

        el_2 = Ellipse(center=(0, 0), rx=4, ry=2)
        data_2 = el_2.evaluate(Grid().get_data(el_2.description, [100, 100]))
        x_2 = data_2[:, 0, np.newaxis]
        y_2 = data_2[:, 1, np.newaxis]
        print(f'MAE: {MAE().evaluate(x_1, x_2):.3f}')
        print(f'RMSE: {RMSE().evaluate(x_1, x_2):.3f}')
