import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from core.data_class import Data
from core.data_description import DataDescription
from core.nets.abstract_net import AbstractNet
from core.data_handling.normalization.normalizer_class import Normalizer
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


class VAE(AbstractNet):
    """
    Вариационный автоэнкодер
    """

    def __init__(self, description: DataDescription,
                 enc_size=[], dec_size=[],
                 epochs=30, batch_size=20, hidden_dim=2):
        """
        :param description: DataDescription. Описание областей определения и значений.
        :param enc_size: List. Список размерностей слоёв энкодера (default=[])
        :param dec_size: List. Список размерностей слоёв декодера (default=[])
        :param epochs: int. Число эпох обучения (default=30)
        :param batch_size: int. Размер батча (default=20)(размер выборки должен быть кратен размеру батча)
        :param hidden_dim: int. Размерность скрытого слоя (default=2)
        """
        assert len(enc_size) == len(dec_size)
        self.desc = description
        self.input_size = description.x_dim + description.y_dim
        self.enc_size = enc_size
        self.dec_size = dec_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.encoder = None  # модель энкодера
        self.decoder = None  # модель декодера
        self.z_mean = None  # математическое ожидание скрытого слоя
        self.z_log_var = None  # дисперсия скрытого слоя
        self.__init_model()

    def __init_model(self):
        def dropout_and_batch(x):
            return Dropout(0.2)(BatchNormalization()(x))

        def create_layers(x, type='enc'):
            if type == 'enc':
                for i in range(len(self.enc_size)):
                    x = Dense(self.enc_size[i], activation='relu')(x)
                    #x = dropout_and_batch(x)
            elif type == 'dec':
                for i in range(len(self.dec_size)):
                    x = Dense(self.dec_size[i], activation='relu')(x)
                    #x = dropout_and_batch(x)
            return x

        input_enc = Input(shape=(self.input_size, 1))
        x = Flatten()(input_enc)
        x = create_layers(x, type='enc')

        self.z_mean = Dense(self.hidden_dim)(x)
        self.z_log_var = Dense(self.hidden_dim)(x)

        def noiser(args):
            self.z_mean, self.z_log_var = args
            N = K.random_normal(shape=(self.batch_size, self.hidden_dim), mean=0., stddev=1.0)
            ex = K.exp(self.z_log_var / 2)
            return ex * N + self.z_mean

        h = Lambda(noiser, output_shape=(self.hidden_dim,))([self.z_mean, self.z_log_var])

        input_dec = Input(shape=(self.hidden_dim,))
        d = create_layers(input_dec, type='dec')
        d = Dense(self.input_size, activation='sigmoid')(d)
        decoded = d

        self.encoder = Model(input_enc, h, name='encoder')
        self.decoder = Model(input_dec, decoded, name='decoder')
        self.model = Model(input_enc, self.decoder(self.encoder(input_enc)), name="vae")

        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'], run_eagerly=True)
        self.model.summary()

    def fit(self, data: Data):
        """
        Обучение нейросети
        :param data: Data. Данные для обучения.
        """
        train_x, test_x, train_y, test_y = train_test_split(data.get_x_norm(),
                                                            data.get_y_norm(),
                                                            test_size=0.2,
                                                            random_state=42)

        train_sample_count = train_x.shape[0] // self.batch_size * self.batch_size
        test_sample_count = test_x.shape[0] // self.batch_size * self.batch_size
        train_x = train_x[:train_sample_count]
        train_y = train_y[:train_sample_count]
        test_x = test_x[:test_sample_count]
        test_y = test_y[:test_sample_count]

        train = np.hstack((train_x, train_y))
        test = np.hstack((test_x, test_y))

        self.model.fit(train, train, validation_data=(test, test),
                       epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

    def loss(self, actual, expected) -> float:
        """
        Функция потерь нейросети
        :param actual: ndarray. Эталонный вектор.
        :param expected: ndarray. Предсказанный вектор.
        :return: float. Ошибка между векторами (скаляр)
        """
        actual = K.reshape(actual, shape=(self.batch_size, self.input_size)).numpy().astype(np.float32)
        expected = K.reshape(expected, shape=(self.batch_size, self.input_size)).numpy().astype(np.float32)
        actual_x = Normalizer.denorm(actual[:, :self.desc.x_dim], self.desc.x_bounds)
        expected_x = Normalizer.denorm(expected[:, :self.desc.x_dim], self.desc.x_bounds)
        actual_y = Normalizer.denorm(actual[:, self.desc.x_dim:], self.desc.y_bounds)
        expected_y = Normalizer.denorm(expected[:, self.desc.x_dim:], self.desc.y_bounds)
        loss = K.sum(K.square(actual_x - expected_x), axis=-1) + K.sum(K.abs(actual_y - expected_y), axis=-1)
        kl_loss = 0.5 * -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return loss + kl_loss

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        Предсказание значения
        :param points: np.ndarray. Точки для предсказания.
        :return: np.ndarray. Предсказанные точки.
        """
        x, y = np.split(points, indices_or_sections=[self.desc.x_dim], axis=1)
        norm_x = Normalizer.norm(x, self.desc.x_bounds)
        norm_y = Normalizer.norm(y, self.desc.y_bounds)
        pred = self.model.predict(np.hstack((norm_x, norm_y)))
        pred_x, pred_y = np.split(pred, indices_or_sections=[self.desc.x_dim], axis=1)
        denorm_x = Normalizer.denorm(pred_x, self.desc.x_bounds)
        denorm_y = Normalizer.denorm(pred_y, self.desc.y_bounds)
        return np.hstack((denorm_x, denorm_y))

    def predict_encoder(self, points: np.ndarray) -> np.ndarray:
        """
        Предсказание значений скрытого слоя
        :param points: np.ndarray. Точки для предсказания.
        :return: np.ndarray. Предсказанные точки скрытого слоя.
        """
        x, y = np.split(points, indices_or_sections=[self.desc.x_dim], axis=1)
        norm_x = Normalizer.norm(x, self.desc.x_dim)
        norm_y = Normalizer.norm(y, self.desc.y_dim)
        pred = self.encoder.predict(np.hstack((norm_x, norm_y)))
        return pred

    def predict_decoder(self,  points: np.ndarray) -> np.ndarray:
        """
        Предсказание значений по скрытому слою
        :param points: np.ndarray. Точки скрытого слоя для предсказания.
        :return: np.ndarray. Предсказанные точки исходного пространства.
        """
        pred = self.decoder.predict(points)
        pred_x, pred_y = np.split(pred, indices_or_sections=[self.desc.x_dim], axis=1)
        denorm_x = Normalizer.denorm(pred_x, self.desc.x_dim)
        denorm_y = Normalizer.denorm(pred_y, self.desc.y_dim)
        return np.hstack((denorm_x, denorm_y))
