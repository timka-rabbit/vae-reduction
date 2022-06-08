from core.optimazation.ego import EGO
from core.optimazation.minimize import ScipyOptimizer
from core.optimazation.brute_force import BruteForce
from functions.abstract_function import AbstractFunc
from core.data_handling.generators.lhs_sequence import LHS
from core.data_handling.generators.sobol_sequence import Sobol
from core.data_description import DataDescription
from core.data_class import Data
from core.metrics.MAE import MAE
from core.metrics.RMSE import RMSE

from core.nets.vae import VAE

from functions.rastrigin_func import Rastrigin
from functions.rosenbrock_func import Rosenbrock

import numpy as np
from unittest import TestCase

MAX_RASTRIGIN = 155.68
MAX_ROSENBROCK = 76826

class TestVAE(TestCase):
    def test_ros(self):
        func = Rosenbrock(n=6)
        max = ScipyOptimizer(func.description).optimize(func.evaluate, n_iter=1000, is_min=False)
        print(func.evaluate(max.reshape(1, -1)))

    def test_ras(self):
        func = Rastrigin(n=6)
        max = ScipyOptimizer(func.description).optimize(func.evaluate, n_iter=2000, is_min=False)
        print(func.evaluate(max.reshape(1, -1)))

    # -------------------------------------------1 эксперимент-------------------------------------------------------- #

    def test_vae(self, func: AbstractFunc, max: float, min:float, irr_dim: int = 0):
        batch_size = 25  # размер батчей
        fit_count = 400  # размерность выборки для обучения
        test_count = 100  # размерность выборки для тестирования
        x_dim = func.description.x_dim  # размерность пространства функции
        dim = x_dim + irr_dim  # полная размерность X
        h_dim = 6

        x = Sobol().get_data(description=func.description, samples_num=fit_count, irrelevant_var_count=irr_dim)
        y = func.evaluate(x)
        data = Data(x, y, DataDescription(dim, func.description.y_dim))
        vae = VAE(description=data.description,
                  enc_size=[], dec_size=[], epochs=15,
                  batch_size=batch_size, hidden_dim=h_dim)

        vae.fit(data)
        vae.save(f'../../../data/net weights/vae/{func.name}_{dim}_{h_dim}.h5')
        # vae.load(f'../../../data/net weights/vae/{func.name}_{dim}_{h_dim}.h5')
        test_x = LHS().get_data(description=func.description, samples_num=test_count, irrelevant_var_count=irr_dim)
        test_y = func.evaluate(test_x)
        test = np.hstack((test_x, test_y))
        err_x = err_y = 0
        for i in range(test_count):
            vae_input = np.repeat(np.expand_dims(test[i].ravel(), axis=0), vae.batch_size, axis=0)
            pred = vae.predict(vae_input)
            for b in range(vae.batch_size):
                pred_x, pred_y = np.split(pred[b], indices_or_sections=[dim], axis=0)
                err_x += MAE().evaluate(test_x[i].reshape(1, dim), pred_x.reshape(1, dim))
                err_y += MAE().evaluate(test_y[i].reshape(1, 1), func.evaluate(pred_x.reshape(1, dim)))
            err_x /= vae.batch_size
            err_y /= vae.batch_size
        err = err_y / test_count
        print(f'MAE(Y, Y\'\'): {err}')
        print(f'Div: {err/(max - min) * 100} %')

    def test_1_vae_rastrigin(self):
        func = Rastrigin(n=6)
        self.test_vae(func=func, max=MAX_RASTRIGIN, min=0, irr_dim=4)

    def test_1_vae_rosenbrock(self):
        func = Rosenbrock(n=6)
        self.test_vae(func=func, max=MAX_ROSENBROCK, min=0, irr_dim=4)

    # -------------------------------------------2 эксперимент-------------------------------------------------------- #

    def test_vae_1_param(self, func: AbstractFunc,  max: float, min:float, irr_dim: int = 0):
        batch_size = 25  # размер батчей
        fit_count = 400  # размерность выборки для обучения
        test_count = 100  # размерность выборки для тестирования
        x_dim = func.description.x_dim  # размерность пространства функции
        dim = x_dim + irr_dim  # полная размерность X
        h_dim = 1

        x_1 = Sobol().get_data(description=DataDescription(x_dim=1, y_dim=1, x_bounds=[func.description.x_bounds[0]]),
                               samples_num=fit_count, irrelevant_var_count=irr_dim)

        def arr(x):
            return np.array([x, x+1, 4*x, 2*x, x+5, 0.5*x])

        x = np.zeros((fit_count, dim))
        for i in range(x.shape[0]):
            x[i] = arr(x_1[i][0])

        y = func.evaluate(x)
        data = Data(x, y, DataDescription(dim, func.description.y_dim))
        vae = VAE(description=data.description,
                  enc_size=[3], dec_size=[3], epochs=15,
                  batch_size=batch_size, hidden_dim=h_dim)

        vae.fit(data)
        vae.save(f'../../../data/net weights/vae/{func.name}_{dim}_{h_dim}.h5')
        # vae.load(f'../../../data/net weights/vae/{func.name}_{dim}_{h_dim}.h5')

        t_1 = Sobol().get_data(description=DataDescription(x_dim=1, y_dim=1, x_bounds=[func.description.x_bounds[0]]),
                               samples_num=test_count, irrelevant_var_count=irr_dim)
        test_x = np.zeros((test_count, dim))
        for i in range(test_x.shape[0]):
            test_x[i] = arr(t_1[i][0])

        test_y = func.evaluate(test_x)
        test = np.hstack((test_x, test_y))
        err_x = err_y = 0
        for i in range(test_count):
            vae_input = np.repeat(np.expand_dims(test[i].ravel(), axis=0), vae.batch_size, axis=0)
            pred = vae.predict(vae_input)
            for b in range(vae.batch_size):
                pred_x, pred_y = np.split(pred[b], indices_or_sections=[dim], axis=0)
                err_x += MAE().evaluate(test_x[i].reshape(1, dim), pred_x.reshape(1, dim))
                err_y += MAE().evaluate(test_y[i].reshape(1, 1), func.evaluate(pred_x.reshape(1, dim)))
            err_x /= vae.batch_size
            err_y /= vae.batch_size
        err = err_y / test_count
        print(f'MAE(Y, Y\'\'): {err}')
        print(f'Div: {err/(max - min) * 100} %')

    def test_2_vae_rastrigin(self):
        func = Rastrigin(n=6)
        self.test_vae_1_param(func=func, max=MAX_RASTRIGIN, min=0, irr_dim=0)

    def test_2_vae_rosenbrock(self):
        func = Rosenbrock(n=6)
        self.test_vae_1_param(func=func, max=MAX_ROSENBROCK, min=0, irr_dim=0)

    # -------------------------------------------3 эксперимент-------------------------------------------------------- #

    def test_vae_search(self, func: AbstractFunc, max: float, min: float,
                        filename: str, irr_dim: int = 0):
        batch_size = 25  # размер батчей
        fit_count = 400  # размерность выборки для обучения
        x_dim = func.description.x_dim  # размерность пространства функции
        dim = x_dim + irr_dim  # полная размерность X
        h_dim = 3

        x = Sobol().get_data(description=func.description, samples_num=fit_count, irrelevant_var_count=irr_dim)
        y = func.evaluate(x)
        data = Data(x, y, DataDescription(dim, func.description.y_dim))
        # vae = VAE(description=data.description,
        #           enc_size=[5], dec_size=[5], epochs=15,
        #           batch_size=batch_size, hidden_dim=h_dim)
        # vae.fit(data)
        # vae.save(f'../../../data/net weights/vae/{func.name}_{dim}_{h_dim}.h5')
        # vae.load(f'../../../data/net weights/vae/{func.name}_{dim}_{h_dim}.h5')
        vae = VAE.create_from_file(description=data.description,
                                   filename=filename)

        def opt_f(x):
            pred = vae.predict_decoder(np.array(x).reshape(1, h_dim))[0]
            pred_x, pred_y = np.split(pred, indices_or_sections=[dim], axis=0)
            return func.evaluate(pred_x.reshape(1, dim))

        h_desc = DataDescription(h_dim, 1, x_bounds=[(0, 1) for i in range(h_dim)])

        # дообучение модели
        # retrain_iter = 50
        # retrain_x = np.zeros((retrain_iter, dim))
        # retrain_y = np.zeros((retrain_iter, func.description.y_dim))

        # for i in range(retrain_iter):
        #     res = BruteForce(h_desc, 10).optimize(func=opt_f)
        #     pred = vae.predict_decoder(np.array(res).reshape((1, h_dim)))[0]
        #     opt_x, opt_y = np.split(pred, indices_or_sections=[dim], axis=0)
        #     retrain_x[i] = opt_x.ravel()
        #     retrain_y[i] = func.evaluate(opt_x).ravel()
        #
        # vae.fine_tune(retrain_x, retrain_y)

        res_hide = BruteForce(h_desc, 10).optimize(func=opt_f)
        pred = vae.predict_decoder(np.array(res_hide).reshape((1, h_dim)))[0]
        opt_x, opt_y = np.split(pred, indices_or_sections=[dim], axis=0)

        res_real = BruteForce(data.description).optimize(func=func.evaluate, n_iter=1000,
                                                         is_grid=False)
        y_opt_real = func.evaluate(res_real).ravel()[0]
        y_opt_hid = func.evaluate(opt_x).ravel()[0]
        print(f'{func.name}')
        print(f'Y\'\' опт: {y_opt_hid}')
        print(f'Div hid: {abs(y_opt_hid - min) / (max - min) * 100} %')
        print(f'Min real Y: {min}')
        print(f'Y опт: {y_opt_real}')
        print(f'Div real: {abs(y_opt_real - min) / (max - min) * 100} %')

    def test_vae_3_rastrigin(self):
        func = Rastrigin(n=6)
        self.test_vae_search(func=func, max=MAX_RASTRIGIN, min=0,
                             filename='Функция Растригина размерности 6_10_3_e_18_s_8.txt',
                             irr_dim=4)

    def test_vae_3_rosenbrock(self):
        func = Rosenbrock(n=6)
        self.test_vae_search(func=func, max=MAX_ROSENBROCK, min=0,
                             filename='Функция Розенброка размерности 6_10_3_e_22_s_5.txt',
                             irr_dim=4)

    # ---------------------------------------------Подбор гиперпараметров--------------------------------------------- #

    def test_vae_ego_parameters(self, func: AbstractFunc, h_dim: int, irr_dim: int = 0):
        n_iter = 10  # количество итераций EGO
        batch_size = 25  # размер батчей
        criterion = 'EI'  # критерий выбора следующей точки EGO
        x_dim = func.description.x_dim  # размерность пространства функции
        fit_count = 400  # размерность выборки для обучения
        test_count = 100  # размерность выборки для тестирования
        dim = x_dim + irr_dim  # полная размерность X

        x = Sobol().get_data(description=func.description, samples_num=fit_count, irrelevant_var_count=irr_dim)
        y = func.evaluate(x)
        data = Data(x, y, DataDescription(dim, func.description.y_dim))

        test_x = LHS().get_data(description=func.description, samples_num=test_count, irrelevant_var_count=irr_dim)
        test_y = func.evaluate(test_x)
        test = np.hstack((test_x, test_y))

        def predict_params(point: np.ndarray) -> np.ndarray:
            """
            :param point: ndarray. Текущая точка оптимизации:
                    point[:, 0] - число эпох
                    point[:, 1] - размер слоёв кодера и декодера
                    point[:, 2] - размер тестовой части
            :return: ndarray. Значение критерия минимизации.
            """
            point_count, n_param = point.shape
            res = np.zeros((point_count, 1))

            for i in range(point_count):
                n_epoch = round(point[i, 0])
                size = round(point[i, 1])
                test_size = point[i, 2]

                vae = VAE(description=data.description,
                          enc_size=[size], dec_size=[size], epochs=n_epoch,
                          batch_size=batch_size, hidden_dim=h_dim, test_size=test_size)
                vae.fit(data)
                vae.save_params(f'{func.name}_{dim}_{h_dim}_e_{n_epoch}_s_{size}.txt')
                err = 0
                for j in range(test_count):
                    vae_input = np.repeat(np.expand_dims(test[j].ravel(), axis=0), vae.batch_size, axis=0)
                    pred = vae.predict(vae_input)
                    for b in range(vae.batch_size):
                        pred_x, pred_y = np.split(pred[b], indices_or_sections=[dim], axis=0)
                        err += MAE().evaluate(test_y[j].reshape(1, 1), func.evaluate(pred_x.reshape(1, dim)))
                    err /= vae.batch_size
                    res[i] += err
                res[i] /= test_count
            return res

        # устанавливаем границы: epoch, size, test_size
        params = DataDescription(x_dim=3, y_dim=1, x_bounds=[(10, 40), (4, 8), (0.05, 0.5)])
        ego = EGO(data_description=params, criterion=criterion)

        # блокировка вывода консоли во время подбора гиперпараметров
        import sys, os
        from contextlib import contextmanager
        import argparse
        @contextmanager
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout
        with suppress_stdout():
            opt_params = ego.optimize(func=predict_params, n_iter=n_iter)

        epochs, size, test_size = opt_params.squeeze()
        print(f'\nEpochs: {round(epochs)}\nEnc/dec size: {round(size)}\nTest size: {test_size}')
        vae = VAE.create_from_file(description=data.description,
                                   filename=f'{func.name}_{dim}_{h_dim}_e_{round(epochs)}_s_{round(size)}.txt')

        err_y = 0
        for i in range(test_count):
            vae_input = np.repeat(np.expand_dims(test[i].ravel(), axis=0), vae.batch_size, axis=0)
            pred = vae.predict(vae_input)
            for b in range(vae.batch_size):
                pred_x, pred_y = np.split(pred[b], indices_or_sections=[dim], axis=0)
                err_y += MAE().evaluate(test_y[i].reshape(1, 1), func.evaluate(pred_x.reshape(1, dim)))
            err_y /= vae.batch_size
        err = err_y / test_count
        print(f'Error: {err}')

    def test_vae_ego_rastrigin(self):
        func = Rastrigin(n=6)
        self.test_vae_ego_parameters(func=func, h_dim=3, irr_dim=4)

    def test_vae_ego_rosenbrock(self):
        func = Rosenbrock(n=6)
        self.test_vae_ego_parameters(func=func, h_dim=3, irr_dim=4)
