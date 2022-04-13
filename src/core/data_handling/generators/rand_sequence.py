import random
import numpy as np

from core.data_handling.generators.abstract_generator import AbstractGenerator
from core.data_description import DataDescription


class RandSeq(AbstractGenerator):
    """
    Случайная генерация данных от 0 до 1
    """
    def get_data(self, description: DataDescription, samples_num: int,
                 irrelevant_var_count: int = 0) -> np.ndarray:
        points = []
        for k in range(samples_num):
            point = []
            for i in description.x_bounds:
                point.append(random.uniform(i[0], i[1]))
            points.append(np.array(point))
        points = self._add_irr_vars(np.array(points), irrelevant_var_count)
        # if write_in_file:
        #     col = [('x' + str(i+1)) for i in range(self.dim + irrelevant_var_count)]
        #     df = pd.DataFrame(arr, columns=col)
        #     df.to_csv(f'../../DataSet/random_{self.dim}_{samples_num}_{irrelevant_var_count}.csv')
        return points
