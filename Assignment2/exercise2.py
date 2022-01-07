import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.optimize import fmin
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def p(xk, alfa, beta):
    return beta / (np.pi * (beta**2 + (xk - alfa)**2))


def pdata(data, alfa, beta):
    value = 1
    for point in data:
        value *= p(point, alfa, beta)
    return value
	
def log_pdata(data, alfa, beta):	
	v = len(data) * np.log(beta/np.pi)
	for point in data:
		v = v - np.log((point - alfa)**2 + beta**2)
	return v
	
def fmin_pdata(x, data):
	a,b = x
	
	value = len(data) * np.log(b/np.pi)
	for point in data:
		value -= np.log((point - a)**2 + b**2)
	return -value


def sample_position():
    alfa = np.random.uniform(0.0, 10.0) # np.random.randint(0, 100)  # between 0 and 10.0 km
    beta = np.random.uniform(1.0, 2.0)  # between 1.0 and 2.0 km
    return alfa, beta


def flashes(alfa, beta):
    return [beta * np.tan(x) + alfa for x in [np.random.uniform(-0.5 * np.pi + 0.001, 0.5 * np.pi - 0.001) for i in range(200)]]


def help(x, flash):
    return sum(flash[:(x + 1)]) / (x + 1)


def likelihood(d):
    fig = plt.figure()
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.arange(-0.0, 5.0, 0.01)
    X, Y = np.meshgrid(x, y)
    ax = fig.gca(projection='3d')

    z = pdata(d, X, Y)
    surf = ax.plot_surface(X, Y, z, cmap=cm.cool, linewidth=0.05, rstride=10, cstride=10)
    #ax.set_zlim(top=4)
    plt.colorbar(surf)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
	
    plt.show()
	
def loglikelihood(d):
    fig = plt.figure()
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.arange(-0.0, 5.0, 0.01)
    X, Y = np.meshgrid(x, y)
    ax = fig.gca(projection='3d')
	
    z = log_pdata(d, X, Y)
    surf = ax.plot_surface(X, Y, z, cmap=cm.cool, linewidth=0.05,rstride=10, cstride=10)
    plt.colorbar(surf)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
	
    plt.show()

def exercise_3_1():
    x = np.linspace(-5, 5, 1000)
    y = pdata([4.8, -2.7, 2.2, 1.1, 0.8, -7.3], x, 1)
    plt.plot(x, y)
    plt.show()
	
def exercise_3_1_2():
	x = np.linspace(-10, 10, 1000)
	y = p(x, 0, 1)
	plt.plot(x, y)
	plt.show()


def exercise_3_2_3(data, alpha, beta):
    x = range(len(data))
    y = [help(x_, data) for x_ in x]
    plt.plot(x, y, label="Running Mean")
    plt.plot(x, [a for _ in x], label="True Alpha")
    plt.legend()

    plt.show()
	
def exercise_3_3_2(data):
    #loglikelihood(data[:1])
    #loglikelihood(data[:2])
    #loglikelihood(data[:3])
    #loglikelihood(data[:20])
        
    #likelihood(data[:1])
    likelihood(data[:2])
	#likelihood(data[:3])
    #likelihood(data[:20])
	
def exercise_3_3_3(data, a, b):
	avals = []
	bvals = []
	for i in range(len(data)):
		f = lambda x: np.log(pdata(data[(i+1)], x[0], x[1]))
		max = fmin(func=fmin_pdata, x0=[0,1], args = (data[:(i+1)],), disp=False)
		avals.append(max[0])
		bvals.append(max[1])
		
	
	plt.plot(range(len(data)), avals, label="Alpha")
	plt.plot(range(len(data)), bvals, label="Beta")
	plt.plot(range(len(data)), [a for i in range(200)], label="True Alpha")
	plt.plot(range(len(data)), [b for i in range(200)], label="True Beta")
	plt.legend()
	axes = plt.gca()
	axes.set_ylim([0, 9])
	plt.show()

np.random.seed(0)
a, b = sample_position()
data = flashes(a,b)
exercise_3_3_2(data)
