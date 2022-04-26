from scipy.stats.qmc import LatinHypercube as LHS_Scipy
import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription
from core.data_handling.normalization.normalizer_class import Normalizer


class LHS(AbstractGenerator):
    """
    Генерация данных методом латинского гиперкуба
    """
    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        sampler = LHS_Scipy(d=description.x_dim)
        points = Normalizer.denorm(sampler.random(samples_num), description.x_bounds)
        points = self._add_irr_vars(description, points, irrelevant_var_count)
        return points
