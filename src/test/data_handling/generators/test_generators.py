from core.data_handling.generators.regular_grid import Grid
from core.data_handling.generators.rand_sequence import RandSeq
from core.data_handling.generators.sobol_sequence import SobolSeq
from core.data_handling.generators.lsh_sequence import LSHSeq

from core.data_description import DataDescription
import ui.chart_utils as ui
from unittest import TestCase


class TestGenerators(TestCase):
    """
    Отрисовка двумерных сгенерированных точек
    """

    def test_grid_2d(self):
        desc = DataDescription(x_dim=2, y_dim=1, x_bounds=[(0, 1), (0, 1)])
        x = Grid().get_data(description=desc, samples_num=1000)
        ui.plot2d_scatter(x[:, 0], x[:, 1], 'Генерация сетки точек')

    def test_rand_2d(self):
        desc = DataDescription(x_dim=2, y_dim=1, x_bounds=[(0, 1), (0, 1)])
        x = RandSeq().get_data(description=desc, samples_num=1000)
        ui.plot2d_scatter(x[:, 0], x[:, 1], 'Случайная генерация точек')

    def test_sobol_2d(self):
        desc = DataDescription(x_dim=2, y_dim=1, x_bounds=[(0, 1), (0, 1)])
        x = SobolSeq().get_data(description=desc, samples_num=1000)
        ui.plot2d_scatter(x[:, 0], x[:, 1], 'Генерация точек Соболем')

    def test_lsh_2d(self):
        desc = DataDescription(x_dim=2, y_dim=1, x_bounds=[(0, 1), (0, 1)])
        x = LSHSeq().get_data(description=desc, samples_num=1000)
        ui.plot2d_scatter(x[:, 0], x[:, 1], 'Генерация точек латинским гиперкубом')

    def test_sobol_3d_irrelevant(self):
        desc = DataDescription(x_dim=2, y_dim=1, x_bounds=[(0, 1), (0, 1)])
        x = SobolSeq().get_data(description=desc, samples_num=1000, irrelevant_var_count=1)
        ui.plot3d_scatter(x[:, 0], x[:, 1], x[:, 2], 'Генерация точек Соболем')
