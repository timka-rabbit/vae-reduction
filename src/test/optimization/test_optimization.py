from core.optimazation.abstract_optimizer import AbstractOptimizer
from core.optimazation.ego import EGO
from core.optimazation.minimize import ScipyOptimizer
from core.optimazation.brute_force import BruteForce

from functions.abstract_function import AbstractFunc
from functions.rosenbrock_func import Rosenbrock
from functions.rastrigin_func import Rastrigin

import numpy as np
from unittest import TestCase


class TestOptimization(TestCase):
    def test_opt_template(self, func: AbstractFunc, optimizer: AbstractOptimizer):
        opt_x = optimizer.optimize(func=func.evaluate, n_iter=100)
        np.set_printoptions(precision=4, floatmode='fixed')
        print(f'{func.name}:')
        print(f'Find X: {opt_x.ravel()}')
        print(f'F(X): {func.evaluate(opt_x).squeeze()}')

    def test_ego_rosenbrock(self):
        func = Rosenbrock()
        self.test_opt_template(func=func, optimizer=EGO(func.description))

    def test_nelder_mead_rosenbrock(self):
        func = Rosenbrock()
        self.test_opt_template(func=func, optimizer=ScipyOptimizer(func.description, method='Nelder-Mead'))

    def test_brute_force_rosenbrock(self):
        func = Rosenbrock()
        self.test_opt_template(func=func, optimizer=BruteForce(func.description, 100))

    def test_ego_rastrigin(self):
        func = Rastrigin(n=2)
        self.test_opt_template(func=func, optimizer=EGO(func.description))

    def test_bfgs_rastrigin(self):
        func = Rastrigin(n=2)
        self.test_opt_template(func=func, optimizer=ScipyOptimizer(func.description, method='BFGS'))

    def test_brute_force_rastrigin(self):
        func = Rastrigin(n=2)
        self.test_opt_template(func=func, optimizer=BruteForce(func.description, 100))
