import sobol_seq
import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription


class SobolSeq(AbstractGenerator):
    """
    Генерация данных от 0 до 1 последовательностью Соболя
    """
    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        points = sobol_seq.i4_sobol_generate(description.x_dim, samples_num)
        points = self._add_irr_vars(points, irrelevant_var_count)
        return points
