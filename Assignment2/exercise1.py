import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

mu_t = [0.45260813, 1.15685635]

'''
Exercise 1:
'''


mu = [1, 0, 1, 2]

matrix = np.matrix("0.14 -0.3 0.0 0.2 ;" +
                   "-0.3 1.16 0.2 -0.8;" +
                   "0.0  0.2  1.0 1.0 ;" +
                   "0.2  -0.8 1.0 2.0")

inverse = np.linalg.inv(matrix)
Sigma_p = matrix[:2, :2] - matrix[:2, 2:] * \
    np.linalg.inv(matrix[2:, 2:]) * matrix[2:, :2]
mu_p = mu[:2] + np.dot(matrix[:2, 2:] * np.linalg.inv(matrix[2:, 2:]),
                       np.subtract([0, 0], mu[2:]))



def part_1():

    # print("Random Numbers: ", np.random.multivariate_normal(mu_p, Sigma_p))

    # print(bivariate_normal(-7, 20, mu_p, Sigma_p))
    def gevaar():
        x, y = np.mgrid[-0.5:2:.01, -0.5:2:.01]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        rv = multivariate_normal(mean=np.ravel(mu_p), cov = Sigma_p)
        # plt.contourf(x, y, rv.pdf(pos))

        # todo, what is sigmaxy?
        # plt.mlab.bivariate_normal(X, Y sigmax sigmay  mux muy sigmaxy)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, rv.pdf(pos), rstride=10, cstride=10,
                        cmap=cm.coolwarm, linewidth=0.05)
		
        plt.colorbar(surf)
        plt.show()
		
    gevaar()


# --------------------------#
# -------   part two -------#
# --------------------------#

def generate(mu):
    pairs = [(x, y) for x, y in [np.random.multivariate_normal(
        mu, np.matrix("2.0 0.8 ; 0.8 4.0")) for _ in range(1000)]]
    np.savetxt('data.txt', pairs)
    return pairs


def part_2():
    pairs = np.loadtxt('data.txt')
    average = (pairs.sum(axis=0) / 1000)
    sum = [[0, 0], [0, 0]]
    for i in range(1000):
        sum += np.dot(np.array([pairs[i] - average]
                               ).transpose(), (np.array([pairs[i] - average])))
    sigmaml = sum / 1000
    TRUESIGMA = sum / (1000 - 1)
    print(sigmaml)
    print(TRUESIGMA)
    print(average)


def part_3():
    pairs = np.loadtxt('data.txt')
    running_average = [np.array([0,0])]
    for i in range(1000):
        running_average.append( (running_average[i] * i + pairs[i]) / (i + 1))
    
    running_average = np.matrix(running_average)
    print(running_average)
    sigma_t = np.matrix("2.0 0.8 ; 0.8 4.0")
    sigma_t_inv = np.linalg.inv(sigma_t)

    def umap(xn, prior_mu, sigman):
        S = np.linalg.inv(np.linalg.inv(sigman) + np.linalg.inv(sigma_t))
        mu = np.dot(S, np.dot(sigma_t_inv, np.array([xn]).transpose()) + np.dot(np.linalg.inv(sigman), prior_mu))
        return (mu, S)


    def initialize_umap():
        Mu = [mu_p.transpose()]
        Sigma = Sigma_p
        for i in range(1000):
            Mu_1,Sigma = umap(pairs[i], Mu[i], Sigma)
            Mu.append(Mu_1)
        return np.array(Mu)

    mu_map = initialize_umap()
    x = [z.transpose().flatten() for z in mu_map]
    
    def plot_things():
        mumaplegend = plt.plot(x)
        avlegend = plt.plot(running_average)
        real = plt.plot([[0.45260813, 1.15685635] for _ in range(1000)], label="real value")
        plt.xlabel("datapoints")
        plt.ylabel("(Estimated) value of mu")

        plt.legend(iter(mumaplegend), ('test',"tessdft"))

        plt.legend(avlegend + mumaplegend + real , ('average 1',"average 2", 'mu_map 1', 'mu_map 2', 'real 1', 'real 2'), loc='lower right')
        plt.show()

    plot_things()




part_3()
# print(generate([1, 1]))

# x = (np.random.multivariate_normal.pdf([1, 1], [[1, 1], [1, 1]]))

# def bivariate_normal(x, y, mean, cov):
#     print(x)
#     x_ = np.array([x, y])
#     normalization = 1 / ((2 * np.pi) *
#                          np.sqrt(np.linalg.det(cov)))
#     exponent = np.dot(np.dot(np.asarray(x_ - mean),
#                              np.linalg.inv(cov)), np.asarray(x - mean))

#     return (normalization * np.exp(-(1 / 2.0) * exponent))[0, 0]
