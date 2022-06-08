import numpy as np

from core.data_class import Data


class AbstractNet(object):
    """
    Абстрактный класс моделей неросетей
    """

    def __init__(self):
        self.model = None

    def fit(self, data: Data):
        """
        Обучение нейросети
        :param data: Data. Данные для обучения.
        """
        NotImplementedError

    def loss(self, actual, expected) -> float:
        """
        Функция потерь нейросети
        :param actual: ndarray. Эталонный вектор.
        :param expected: ndarray. Предсказанный вектор.
        :return: float. Ошибка между векторами (скаляр)
        """
        NotImplementedError

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        Предсказание значения
        :param points: np.ndarray. Точки для предсказания.
        :return: np.ndarray. Предсказанные точки.
        """
        NotImplementedError

    def save(self, filename: str):
        """
        Сохранение весов модели
        :param filename: str. Имя файла для сохранения.
        """
        NotImplementedError

    def load(self, filename: str):
        """
        Загрузка весов модели
        :param filename: str. Имя файла для загрузки.
        """
        NotImplementedError

    def save_params(self, filename: str):
        """
        Сохранение гиперпараметров модели
        :param filename: str. Имя файла для сохранения.
        """
        NotImplementedError
