from core.optimazation.ego import EGO
from core.optimazation.minimize import Minimizer
from functions.abstract_function import AbstractFunc
from core.data_handling.generators.lhs_sequence import LHS
from core.data_handling.generators.sobol_sequence import Sobol
from core.data_description import DataDescription
from core.data_class import Data
from core.metrics.MAE import MAE

from core.nets.vae import VAE

from functions.rastrigin_func import Rastrigin

import numpy as np
from unittest import TestCase


class TestVAE(TestCase):
    def test_vae(self, func: AbstractFunc, irr_dim: int = 0):
        batch_size = 25  # размер батчей
        fit_count = 10000  # размерность выборки для обучения
        test_count = 200  # размерность выборки для тестирования
        x_dim = func.description.x_dim  # размерность пространства функции
        dim = x_dim + irr_dim  # полная размерность X

        x = Sobol().get_data(description=func.description, samples_num=fit_count, irrelevant_var_count=irr_dim)
        y = func.evaluate(x)
        data = Data(x, y, DataDescription(dim, func.description.y_dim))
        vae = VAE(description=data.description,
                  enc_size=[4], dec_size=[4], epochs=30,
                  batch_size=batch_size, hidden_dim=2)
        vae.fit(data)
        vae.save(f'../../../data/net weights/vae/{func.name}_{dim}_{2}.h5')
        # vae.load(f'../../../data/net weights/vae/{func.name}_{dim}_{2}.h5')
        test_x = LHS().get_data(description=func.description, samples_num=test_count, irrelevant_var_count=irr_dim)
        test_y = func.evaluate(test_x)
        test = np.hstack((test_x, test_y))
        res = 0
        iter = test_count//batch_size
        for i in range(0, test_count, batch_size):
            pred = vae.predict(test[i:i+batch_size, :])
            pred_x, pred_y = np.split(pred, indices_or_sections=[dim], axis=1)
            res += MAE().evaluate(test_y[i:i+batch_size, :], pred_y)
        print(f'MAE: {res / iter}')

    def test_vae_rastrigin(self):
        func = Rastrigin(n=6)
        self.test_vae(func=func, irr_dim=0)

    # ---------------------------------------------------------------------------------------------------------------- #

    def test_vae_ego_parameters(self, func: AbstractFunc, irr_dim: int = 0):
        n_iter = 10  # количество итераций EGO
        batch_size = 25  # размер батчей
        criterion = 'EI'  # критерий выбора следующей точки EGO
        x_dim = func.description.x_dim  # размерность пространства функции
        fit_count = 10000  # размерность выборки для обучения
        test_count = 200  # размерность выборки для тестирования
        dim = x_dim + irr_dim  # полная размерность X

        x = Sobol().get_data(description=func.description, samples_num=fit_count, irrelevant_var_count=irr_dim)
        y = func.evaluate(x)
        data = Data(x, y, DataDescription(dim, func.description.y_dim))

        test_x = LHS().get_data(description=func.description, samples_num=test_count, irrelevant_var_count=irr_dim)
        test_y = func.evaluate(test_x)

        def predict_params(point: np.ndarray) -> np.ndarray:
            """
            :param point: ndarray. Текущая точка оптимизации:
                    point[:, 0] - число эпох
                    point[:, 1] - размер скрытого слоя
            :return: ndarray. Значение критерия минимизации.
            """
            point_count, n_param = point.shape
            res = np.zeros((point_count, 1))

            for i in range(point_count):
                n_epoch = int(point[i, 0])
                h_dim = int(point[i, 1])

                vae = VAE(description=data.description,
                          enc_size=[4], dec_size=[4], epochs=n_epoch,
                          batch_size=batch_size, hidden_dim=h_dim)
                vae.fit(data)
                vae.save(f'../../../data/net weights/vae/{func.name}_ego_{dim}_{h_dim}.h5')
                for j in range(vae.batch_size, test_count, vae.batch_size):
                    pred_x = vae.predict(test_x[j-batch_size:j, :])
                    pred_y = func.evaluate(pred_x)
                    res[i] += MAE().evaluate(test_y[j-batch_size:j, :], pred_y)
                res[i] /= (test_count / batch_size)
            return res

        # устанавливаем границы: epoch, batch_size, hidden_dim
        params = DataDescription(x_dim=3, y_dim=1, x_bounds=[(10, 40), (1, 3)])
        ego = EGO(data_description=params, criterion=criterion)
        opt_params = ego.optimize(func=predict_params, n_iter=n_iter)

        epochs, hidd = opt_params.squeeze()
        print(f'\nEpochs: {int(epochs)}\nHidden dim: {int(hidd)}')
        vae = VAE(func=func, enc_size=[4], dec_size=[4], epochs=int(epochs),
                  batch_size=batch_size, hidden_dim=int(hidd))
        vae.load(f'../../../data/net weights/vae/{func.name}_ego_{x_dim+irr_dim}_{int(hidd)}.h5')

        err = 0
        for j in range(vae.batch_size, test_count, vae.batch_size):
            pred_x = vae.predict(test_x[j - batch_size:j, :])
            pred_y = func.evaluate(pred_x)
            err += MAE().evaluate(test_y[j - batch_size:j, :], pred_y)
        err /= (test_count / batch_size)
        print(f'Error: {err}')

    def test_vae_ego_rastrigin(self):
        func = Rastrigin(n=6)
        self.test_vae_ego_parameters(func=func, irr_dim=0)
