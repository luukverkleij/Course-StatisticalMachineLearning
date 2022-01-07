import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

np.random.seed(31415)
alpha = 2
beta = 10
x_data = np.array([0.4, 0.6])
t_data = np.array([0.6, -0.35])
string = str(alpha) + " 0; 0 " + str(alpha) + ""

S_n1 = np.matrix(string)  + \
    2 * beta * np.matrix(" 1 %f ; %f %f" %
                         (np.mean(x_data), np.mean(x_data), np.mean(np.square(x_data))))
M_n = 2 * beta * np.dot(np.linalg.inv(S_n1), np.matrix([np.mean(t_data), np.dot(x_data, t_data) / 2]).T)

print(M_n)
def phi(x):
    return np.array([1, x]).transpose()


def s_squared(x):
    return 1 / beta + np.dot(np.dot(phi(x).transpose(), np.linalg.inv(S_n1)), phi(x))


def m(x):
    
    return np.dot(phi(x).T, M_n)


def sample():
    return (np.random.multivariate_normal(np.ravel(M_n.transpose()), np.linalg.inv(S_n1)))

sample()

def printshitopplaatje():
    x = np.arange(0,1,0.001)

    mean = [(m(xi).mean()) for xi in x]
    sigma_up = [(m(xi).mean() + s_squared(xi).mean())  for xi in x]
    sigma_down = [(m(xi).mean() - s_squared(xi).mean())  for xi in x]

    ax = plt.gca()
    ax.set_ylim(-0.4,0.8)
    ax.set_xlim(0.0, 1.0)
    plt.fill_between(x,sigma_down,sigma_up, facecolor="#F012BE", alpha=0.5, label="Standard deviation")
    plt.plot(x,mean, color="#85144b", label="Mean", linewidth=2)
    plt.scatter(x_data, t_data, color="red", s=100, label="Data points")

    for i in range(5):
        weights = sample()
        F = weights[0] + weights[1]*x
        if i == 0:
            plt.plot(x,F, color="#001f3f", label="Sampled $y(x, w)$")
        else:
            plt.plot(x,F, color="#001f3f")


    plt.legend(scatterpoints=1)
    plt.show()

printshitopplaatje()