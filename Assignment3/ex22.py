import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ssp


x = np.loadtxt('data')
alldata = x
xs = np.asarray([[y[0], y[1]] for y in x if y[2] < 0.5])
ys = np.array([[y[0], y[1]] for y in x if y[2] >= 0.5])

w_0 = np.array([0, 0, 0])

t_data = np.array([p[2] for p in x])


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def cross_entropy(w, phi):
	y = sigmoid(np.dot(w.T, phi.T))
	s = 0
	for i in range(len(y)):
		s += t_data[i] * np.log(y[i]) + (1-t_data[i])*np.log(1-y[i])
	return -s


def gE(w, phi):
    y = sigmoid(np.dot(w.T, phi.T))
    return np.dot(phi.T, (y - t_data))


def ggE(w, phi):
    ys = sigmoid(np.dot(w.T, phi.T))
    R = np.diag([(y * (1 - y)) for y in ys])

    return np.dot(np.dot(phi.T, R), phi)


def weight(w,phi):
    return w - np.dot(np.linalg.inv(ggE(w, phi)), gE(w, phi))

def cal_weight(w, phi, count):
    for _ in range(count):
        w = weight(w, phi)
    return w


def plot_scatter(w,phi):
    colors = sigmoid(np.dot(w.T, phi.T))
    plt.scatter(alldata.T[0],alldata.T[1], c=colors, cmap="RdBu_r", label="scatter",s=100)
    plt.colorbar()

# plot_scatter()
# plt.show()



###
def gaus(point, mean):
	s = np.matrix('0.2 0; 0 0.2')
	return ssp.multivariate_normal.pdf(point, mean, s)

def exercise22_1():
    plt.plot(xs.T[0], xs.T[1], 'o')
    plt.plot(ys.T[0], ys.T[1], 'o', color='red')
    plt.show()

def exercise22_3():
    phi = np.array([[1, p[0], p[1]] for p in x])
    w = cal_weight(w_0, phi, 5)
    plot_scatter(w, phi)
    plt.show()
	
def exercise22_4():
	appelsap = np.asarray([[y[0], y[1]] for y in x])
	phi1 = gaus(appelsap,[0,0])
	phi2 = gaus(appelsap,[1,1])
	phi = np.array([[1,x[0],x[1]] for x in zip(phi1,phi2)])
	
	xs = np.array([[y[0],y[1]] for y in zip(phi1,phi2,t_data) if y[2] < 0.5])
	ys = np.array([[y[0],y[1]] for y in zip(phi1,phi2,t_data) if y[2] >= 0.5])
	
	plt.plot(xs.T[0], xs.T[1], 'o')
	plt.plot(ys.T[0], ys.T[1], 'o', color='red')
	plt.show()
	
def exercise22_5():
	appelsap = np.asarray([[y[0], y[1]] for y in x])
	phi1 = gaus(appelsap,[0,0])
	phi2 = gaus(appelsap,[1,1])
	phi = np.array([[1,x[0],x[1]] for x in zip(phi1,phi2)])
	
	xs = np.array([[y[0],y[1]] for y in zip(phi1,phi2,t_data) if y[2] < 0.5])
	ys = np.array([[y[0],y[1]] for y in zip(phi1,phi2,t_data) if y[2] >= 0.5])
	
	w = cal_weight(w_0, phi, 5)
	plot_scatter(w, phi)
	plt.show()

exercise22_3()