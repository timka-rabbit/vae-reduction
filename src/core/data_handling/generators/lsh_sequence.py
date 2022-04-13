from smt.sampling_methods import LHS
import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription


class LSHSeq(AbstractGenerator):
    """
    Генерация данных методом латинского гиперкуба
    """
    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        x_limits = np.array(description.x_bounds)
        sampling = LHS(xlimits=x_limits, criterion='corr')
        points = sampling(samples_num)
        points = self._add_irr_vars(points, irrelevant_var_count)
        return points
