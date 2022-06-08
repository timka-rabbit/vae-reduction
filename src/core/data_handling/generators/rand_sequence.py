import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription
from core.data_handling.normalization.normalizer_class import Normalizer


class Rand(AbstractGenerator):
    """
    Случайная генерация данных
    """

    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        points = np.zeros((samples_num, description.x_dim))
        for i in range(samples_num):
            for j in range(description.x_dim):
                points[i, j] = np.random.uniform(0, 1)
        points = Normalizer.denorm(points, description.x_bounds)
        points = self._add_irr_vars(np.array(points), irrelevant_var_count)
        return points
