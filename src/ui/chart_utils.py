import matplotlib.pyplot as plt
import numpy as np


def plot2d(x, y, title=''):
    """
    Отрисовка 2D графиков
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, y)
    ax.set_title(label=title)
    ax.set_aspect(1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def plot3d(x, y, z, title=''):
    """
    Отрисовка 3D графиков
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    n = int(x.shape[0]**(1/2))
    ax.plot_surface(x.reshape(n, n), y.reshape(n, n), z.reshape(n, n), cmap=plt.cm.Spectral)
    ax.set_title(label=title)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot2d_scatter(x, y, title=''):
    """
    Отрисовка точечных 2D графиков
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y)
    ax.set_title(label=title)
    ax.set_aspect(1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def plot3d_scatter(x, y, z, title=''):
    """
    Отрисовка точечных 3D графиков
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='^')
    ax.set_title(label=title)
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
