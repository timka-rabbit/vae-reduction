import os
import numpy as np


def read_data_from_csv(filepath, absolute_path=False):
    """
    Чтение табличных числовых данных в двумерный массив

    :param filepath: str. Название файла.
    :param absolute_path: str. Абсолютный путь файла.
    :return: Массив данных.
    """
    try:
        if not absolute_path:
            filepath = get_absolute_path(filepath)
        return np.genfromtxt(filepath, delimiter=',')
    except OSError as e:
        raise OSError(e.args[0])


def write_data_to_csv(filepath, data, absolute_path=False):
    """
    Запись табличных числовых данных в файл

    :param filepath: str. Название файла.
    :param data: ndarray. Массив данных.
    :param absolute_path: str. Абсолютный путь файла.
    """
    try:
        if not absolute_path:
            filepath = get_absolute_path(filepath)
        np.savetxt(filepath, data, delimiter=',')
    except OSError as e:
        raise OSError(e.args[0])


def get_absolute_path(relative_path):
    """
    Вычисление абсолютного пути файла относительно корня проекта

    :param relative_path: str. Относительный путь файла.
    """
    return os.path.join(os.path.dirname(__file__), '../../' + relative_path)
