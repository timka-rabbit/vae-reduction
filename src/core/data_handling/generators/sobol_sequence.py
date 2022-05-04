from scipy.stats.qmc import Sobol as Sobol_Scipy
import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription
from core.data_handling.normalization.normalizer_class import Normalizer


class Sobol(AbstractGenerator):
    """
    Генерация данных последовательностью Соболя
    """
    def __init__(self, seed=2):
        self.seed = seed

    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        sampler = Sobol_Scipy(d=description.x_dim, seed=self.seed)
        points = Normalizer.denorm(sampler.random(samples_num), description.x_bounds)
        points = self._add_irr_vars(description, points, irrelevant_var_count)
        return points
