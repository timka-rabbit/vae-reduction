from scipy.stats.qmc import LatinHypercube as LSH
import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription


class LSHSeq(AbstractGenerator):
    """
    Генерация данных методом латинского гиперкуба
    """
    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        sampler = LSH(d=description.x_dim)
        points = sampler.random(samples_num)
        points = self._add_irr_vars(points, irrelevant_var_count)
        return points
