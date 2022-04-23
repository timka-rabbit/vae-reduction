import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
#np.seterr(divide='ignore', invalid='ignore')
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from core.data_class import Data
from functions.abstract_function import AbstractFunc
from core.nets.abstract_net import AbstractNet
from core.data_handling.normalization.normalizer_class import Normalizer


class VAE(AbstractNet):
    """
    Вариационный автоэнкодер
    """

    def __init__(self, func: AbstractFunc, layers=0, enc_size=[], dec_size=[],
                 epochs=30, batch_size=20, hidden_dim=2):
        """
        :param func: AbstractFunc. Функция для обучения.
        :param layers: int. Количество слоёв энкодера и декодера между входным и скрытым (default=2)
        :param enc_size: List. Список размерностей слоёв энкодера (default=[])
        :param dec_size: List. Список размерностей слоёв декодера (default=[])
        :param epochs: int. Число эпох обучения (default=30)
        :param batch_size: int. Размер батча (default=20)(размер выборки должен быть кратен размеру батча)
        :param hidden_dim: int. Размерность скрытого слоя (default=2)
        """
        assert layers == len(enc_size) == len(dec_size)
        self.input_size = func.description.x_dim
        self.func = func
        self.layers = layers
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.encoder = None
        self.decoder = None
        self.z_mean = None  # матожидание скрытого слоя
        self.z_log_var = None  # дисперсия скрытого слоя

    def fit(self, data: Data):
        """
        Обучение нейросети
        :param data: Data. Данные для обучения.
        """
        norm_data = Normalizer.norm(data)
        train_x, test_x, train_y, test_y = train_test_split(norm_data.x,
                                                            norm_data.y,
                                                            test_size=0.2,
                                                            random_state=1)

        def dropout_and_batch(x):
            return Dropout(0.3)(BatchNormalization()(x))

        def create_layers(x, type='enc'):
            if type == 'enc':
                for i in range(1, self.layers):
                    x = Dense(self.enc_size[i], activation='relu')(x)
                    x = dropout_and_batch(x)
            elif type == 'dec':
                for i in range(0, self.layers - 1):
                    x = Dense(self.dec_size[i], activation='relu')(x)
                    x = dropout_and_batch(x)
            return x

        input_enc = Input((self.input_size, 1))
        x = Flatten()(input_enc)
        x = create_layers(x, type='enc')

        self.z_mean = Dense(self.hidden_dim)(x)
        self.z_log_var = Dense(self.hidden_dim)(x)

        def noiser(args):
            self.z_mean, self.z_log_var = args
            N = K.random_normal(shape=(self.batch_size, self.hidden_dim), mean=0., stddev=1.0)
            return K.exp(self.z_log_var / 2) * N + self.z_mean

        h = Lambda(noiser, output_shape=(self.hidden_dim,))([self.z_mean, self.z_log_var])

        input_dec = Input(shape=(self.hidden_dim,))
        d = create_layers(input_dec, type='dec')
        d = Dense(self.input_size, activation='sigmoid')(d)
        decoded = d

        self.encoder = Model(input_enc, h, name='encoder')
        self.decoder = Model(input_dec, decoded, name='decoder')
        self.model = Model(input_enc, self.decoder(self.encoder(input_enc)), name="vae")

        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])
        self.model.summary()

        self.model.fit(train_x, train_x, validation_data=(test_x, test_x),
                       epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

    def loss(self, actual, expected) -> float:
        """
        Функция потерь нейросети
        :param actual: ndarray. Эталонный вектор.
        :param expected: ndarray. Предсказанный вектор.
        :return: float. Ошибка между векторами (скаляр)
        """
        actual = K.reshape(actual, shape=(self.batch_size, self.input_size))
        expected = K.reshape(expected, shape=(self.batch_size, self.input_size))
        loss = K.sum(K.square(self.func.evaluate(actual) - self.func.evaluate(expected)), axis=-1)
        kl_loss = 0.05 * -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return loss + kl_loss

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        Предсказание значения
        :param points: np.ndarray. Точки для предсказания.
        :return: np.ndarray. Предсказанные точки.
        """
        return self.model.predict(points)

    def predict_encoder(self, points: np.ndarray) -> np.ndarray:
        """
        Предсказание значений скрытого слоя
        :param points: np.ndarray. Точки для предсказания.
        :return: np.ndarray. Предсказанные точки скрытого слоя.
        """
        return self.encoder.predict(points)

    def predict_decoder(self,  points: np.ndarray) -> np.ndarray:
        """
        Предсказание значений по скрытому слою
        :param points: np.ndarray. Точки скрытого слоя для предсказания.
        :return: np.ndarray. Предсказанные точки исходного пространства.
        """
        return self.decoder.predict(points)
