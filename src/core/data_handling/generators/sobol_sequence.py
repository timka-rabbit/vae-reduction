from scipy.stats.qmc import Sobol
import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription


class SobolSeq(AbstractGenerator):
    """
    Генерация данных от 0 до 1 последовательностью Соболя
    """
    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        sampler = Sobol(d=description.x_dim)
        points = sampler.random(samples_num)
        points = self._add_irr_vars(points, irrelevant_var_count)
        return points
