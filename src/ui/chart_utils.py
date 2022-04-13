import matplotlib.pyplot as plt


def plot2d_scatter(x, y, title=''):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x, y)
    ax.set_title(label=title)
    plt.show()
