from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np


def h(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2


def dhx(x, y):
    return 2 * (-1 + x + 200 * x**3 - 200 * x * y)


def dhy(x, y):
    return 200 * (y - x**2)


def descent_next(w_x, w_y, epsilon):
    return ((-epsilon * dhx(w_x, w_y)), (-epsilon * dhy(w_x, w_y)))


def descent(w_x, w_y, epsilon, precision):
    while True:
        dn = descent_next(w_x, w_y, epsilon)
        w_x = w_x + dn[0]
        w_y = w_y + dn[1]
        if np.sqrt((dn[0] / epsilon)**2 + (dn[1] / epsilon)**2) < precision:
            break

    return (w_x, w_y)


def plot_descent(w_x, w_y, epsilon, precision, color="blue"):
    start_x = w_x
    start_y = w_y

    xs = [w_x]
    ys = [w_y]
    tel = 0
    while True:
        tel += 1
        dn = descent_next(w_x, w_y, epsilon)
        w_x = w_x + dn[0]
        w_y = w_y + dn[1]

# if not xs or np.sqrt((w_x - xs[-1])**2 + (w_y - ys[-1])**2) > 0.05:
        xs += [w_x]
        ys += [w_y]

        if np.sqrt((dn[0] / epsilon)**2 + (dn[1] / epsilon)**2) < precision:
            break

    xs += [w_x]
    ys += [w_y]
    print(tel)
    #plt.scatter(xs, ys, color=color)
    plt.plot(xs, ys, linewidth=2, color=color,
             label="(" + str(start_x) + "," + str(start_y) + ")")


def plot_surface():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(-2.0, 2.0, 0.25)
    y = np.arange(-1.0, 3.0, 0.25)
    X, Y = np.meshgrid(x, y)
    Z = h(X, Y)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_zlim(-2, 3000)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)


def plot_contour():
    x = np.arange(-2.0, 2.0, 0.001)
    y = np.arange(-1.0, 3.0, 0.001)
    X, Y = np.meshgrid(x, y)
    Z = h(X, Y)
    plt.figure()
    levels = [0, 0.1, 1, 10, 100, 400, 1200]
    CS = plt.contour(X, Y, Z, levels, colors=('k'))
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Simplest default with labels')
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    l, b, w, h1 = plt.gca().get_position().bounds
    ll, bb, ww, hh = CB.ax.get_position().bounds
    CB.ax.set_position([ll, b + 0.1 * h1, ww, h1 * 0.8])


def assigment2_1():
    plot_surface()
    plt.show()


def assigment2_4():
    plot_contour()

    #plot_descent(-2, -1, 0.0015, 0.000000001, "blue")
    plot_descent(0, 3, 0.0015, 0.000000001, "green")
    plot_descent(2, 3, 0.0015, 0.000000001, "red")
    plt.legend(bbox_to_anchor=(0., 0.25, 1., 0))
    plt.show()

assigment2_1()
