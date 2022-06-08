from functions.abstract_function import AbstractFunc
from core.data_handling.generators.lhs_sequence import LHS
from core.data_handling.generators.sobol_sequence import Sobol
from core.data_class import Data

from core.nets.vae import VAE
from core.nets.ae import AE

from functions.rastrigin_func import Rastrigin
from functions.rosenbrock_func import Rosenbrock

import numpy as np
import ui.chart_utils as ui
from unittest import TestCase


class TestAE(TestCase):
    def test_hidden_dim(self, func: AbstractFunc):
        batch_size = 20  # размер батчей
        fit_count = 400  # размерность выборки для обучения
        test_count = 400  # размерность выборки для тестирования
        x_dim = func.description.x_dim  # размерность пространства функции
        h_dim = 2

        x = Sobol().get_data(description=func.description, samples_num=fit_count, irrelevant_var_count=0)
        y = func.evaluate(x)
        data = Data(x, y, func.description)

        ae = AE(description=data.description,
                enc_size=[], dec_size=[], epochs=15,
                batch_size=batch_size, hidden_dim=h_dim)
        ae.fit(data)
        ae.save(f'../../../data/net weights/ae/{func.name}_{x_dim}_{h_dim}.h5')

        vae = VAE(description=data.description,
                  enc_size=[], dec_size=[], epochs=15,
                  batch_size=batch_size, hidden_dim=h_dim)
        vae.fit(data)

        test_x = LHS().get_data(description=func.description, samples_num=test_count, irrelevant_var_count=0)
        test_y = func.evaluate(test_x)
        test = np.hstack((test_x, test_y))

        h_points_ae = np.zeros((test_count, h_dim))
        h_points_vae = np.zeros((test_count, h_dim))
        from core.data_handling.normalization.normalizer_class import Normalizer
        for i in range(test_count):
            h_points_ae[i] = ae.predict_encoder(test[i].reshape(1, -1))
            vae_input = np.repeat(np.expand_dims(test[i].ravel(), axis=0), vae.batch_size, axis=0)
            h_points_vae[i] = Normalizer.norm(vae.predict_encoder(vae_input), [(-3, 3)]*2)[0]

        ui.plot2d_scatter(h_points_ae[:, 0], h_points_ae[:, 1], title='Скрытое пространство автоэнкодера')
        ui.plot2d_scatter(h_points_vae[:, 0], h_points_vae[:, 1], title='Скрытое пространство вариационного автоэнкодера')

    def test_rastringin(self):
        func = Rastrigin(n=6)
        self.test_hidden_dim(func)

    def test_rosenbrock(self):
        func = Rosenbrock(n=6)
        self.test_hidden_dim(func)
