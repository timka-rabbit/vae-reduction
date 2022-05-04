from core.optimazation.ego import EGO
from core.optimazation.minimize import Minimizer
from functions.abstract_function import AbstractFunc
from core.data_handling.generators.rand_sequence import Rand
from core.data_handling.generators.sobol_sequence import Sobol
from core.data_description import DataDescription
from core.data_class import Data
from core.metrics.MAE import MAE

from core.nets.vae import VAE

from functions.rastrigin_func import Rastrigin

import numpy as np
from unittest import TestCase


class TestVAE(TestCase):
    def test_vae_ego_parameters(self, func: AbstractFunc, irr_dim: int = 0):
        n_iter = 25  # количество итераций EGO
        criterion = 'EI'  # критерий выбора следующей точки EGO
        x_dim = func.description.x_dim  # размерность пространства функции
        fit_count = 10000  # размерность выборки для обучения
        test_count = 200  # размерность выборки для тестирования

        x = Sobol().get_data(description=func.description, samples_num=fit_count, irrelevant_var_count=irr_dim)
        y = func.evaluate(x)
        data = Data(x, y, DataDescription(func.description.x_dim+irr_dim, func.description.y_dim))

        test_x = Rand().get_data(description=func.description, samples_num=test_count, irrelevant_var_count=irr_dim)
        test_y = func.evaluate(test_x)

        def predict_params(point: np.ndarray) -> np.ndarray:
            """
            :param point: ndarray. Текущая точка оптимизации:
                    point[:, 0] - число эпох
                    point[:, 1] - батчсайз
                    point[:, 2] - размер скрытого слоя
            :return: ndarray. Значение критерия минимизации.
            """
            point_count, n_param = point.shape
            res = np.zeros((point_count, 1))

            for i in range(point_count):
                n_epoch = int(point[i, 0])
                b_size = int(point[i, 1])
                h_dim = int(point[i, 2])

                while int(fit_count*0.8) % b_size != 0:
                    b_size -= 1

                vae = VAE(description=data.description, func=func,
                          layers=1, enc_size=[4], dec_size=[4], epochs=n_epoch,
                          batch_size=b_size, hidden_dim=h_dim)
                vae.fit(data)
                vae.save('../../../data/net weights/vae/' +
                         f'{func.name}_ego_{func.description.x_dim+irr_dim}_{h_dim}.h5')
                pred_x = vae.predict(test_x)
                pred_y = func.evaluate(pred_x)
                res[i] = MAE().evaluate(test_y, pred_y)
            return res

        # устанавливаем границы: epoch, batch_size, hidden_dim
        params = DataDescription(x_dim=3, y_dim=1, x_bounds=[(10, 40), (20, 64), (1, 3)])
        ego = EGO(data_description=params, criterion=criterion)
        opt_params = ego.optimize(func=predict_params, n_iter=n_iter)

        epochs, batch, hidd = opt_params.squeeze()
        print(f'\nEpochs: {int(epochs)}\nBatch size: {int(batch)}\nHidden dim: {int(hidd)}')
        vae = VAE(func=func, layers=1, enc_size=[4], dec_size=[4], epochs=int(epochs),
                  batch_size=int(batch), hidden_dim=int(hidd))
        vae.load(f'../../../data/net weights/vae/{func.name}_ego_{func.description.x_dim+irr_dim}_{int(hidd)}.h5')
        pred_x = vae.predict(test_x)
        pred_y = func.evaluate(pred_x)
        err = MAE().evaluate(test_y, pred_y)
        print(f'Error: {err}')

    def test_vae_rastrigin(self):
        func = Rastrigin(n=6)
        self.test_vae_ego_parameters(func=func, irr_dim=0)
