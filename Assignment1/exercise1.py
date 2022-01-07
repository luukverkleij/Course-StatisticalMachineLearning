import numpy as np
import matplotlib.pyplot as plt

'''
Exercise 1: 
'''
def f(x):
    return 1 + np.sin(6 * (x - 2))

#t = np.arange(0.0, 1.0, 0.01)
#t2 = np.arange(0.0, 1.0, 0.1)
#t3 = np.arange(0.0, 1.0, 0.025)

np.random.seed(0)

def createNoise(n):
    return np.random.normal(0, 0.3, n)


def createSignal(n):
    t = np.arange(0.0, 1, 1 / n)
    return (t, [x + y for x, y in zip(createNoise(n), [f(x) for x in t])])

t10 = np.arange(0.0, 1, 0.11111111111111)
print(len(t10))
s10 = (t10, [x + y for x, y in zip(createNoise(10), [f(x) for x in t10])])
s40 = createSignal(40)
s100 = createSignal(100)


#testset = [x + y for x, y in zip(createNoise(100), [f(x) for x in t])]
#dataset = [x +  y for x, y in zip(createNoise(10), [f(x) for x in t2])]
#dataset40 = [x + y for x, y in zip(createNoise(40), [f(x) for x in t3])]


def sumx(x, i):
    j = 0
    s = 0
    while(j < len(x)):
        s += x[j]**i
        j += 1
    return s


def sumy(x, t, i):
    j = 0
    s = 0
    while(j < len(x)):
        s += t[j] * x[j]**i
        j += 1
    return s


def kdelta(i, j):
    # works due to pythons representation of true (1) and false (0)
    return i == j

'''
Exercise 1_2: E(w) = 0.5 * SUM(N, n=1, {y(xn; w) − tn}^2)
2

'''
def PolCurFit(x, t, m):
    # Exercise 2
    # D_n = (x, t)
    A = np.fromfunction(lambda i, j: sumx(x, i + j), (m + 1, m + 1), dtype=int)
    T = np.fromfunction(lambda i, j: sumy(x, t, i), (m + 1, 1), dtype=int)
    return np.linalg.solve(A, T)

'''
Exercise 1_5: E˜ = E + λ/2 * SUM(M, j=0, w[j]^2)
'''
def PolCurFit2(x, t, m, λ):
    # Exercise 5
    # Sum_j=0^M Aij * wj = Tj
    # Based on exercise sheet 2, exercise 2.3
    A = np.fromfunction(
        lambda i, j: sumx(x, i + j) + λ * kdelta(i, j),
        (m + 1, m + 1),
        dtype=int)
    T = np.fromfunction(lambda i, j: sumy(x, t, i), (m + 1, 1), dtype=int)
    return np.linalg.solve(A, T)


def poly(vec, x):
    s = 0
    j = 0
    while(j < len(vec)):
        s += x**j * vec[j]
        j += 1
    return s

def error_rms(w, ts, ds):
    E = 0
    n = len(ts)
    for j in range(0, n):
        E += (poly(w.flatten(), ts[j]) - ds[j])**2
    rms = np.sqrt(E/n)

    return rms


'''
Functions to execute the exercises.
'''

def exercise1_1():
    plt.plot(s100[0], f(s100[0]), label=("f"))
    plt.plot(s10[0], s10[1], 'ro', label=("dataset 10"))
    plt.legend()
    plt.show()

def exercise_a(ts, ds):
    for i in range(0, 10):
        w = PolCurFit(ts, ds, i)
        x = np.linspace(0, 1, 100)
        y = poly(w.flatten(), x)
        plt.plot(x, y, linestyle="dashed", label=("M=" + str(i)))
    plt.plot(ts, ds, 'ro')
    plt.plot(s100[0], f(s100[0]), linewidth=1.5, label=("f"))
    plt.legend(bbox_to_anchor=(0., 0.85, 1., 0), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
    plt.gca().set_ylim([-1,4])

    plt.show()

def exercise_b(ts, ds):
    e_train = []
    e_test = []
    for i in range(0, 10):
        w = PolCurFit(ts, ds, i)
        e_train += [error_rms(w, ts, ds)]
        e_test += [error_rms(w, s100[0], s100[1])]
    
    
    plt.plot(np.arange(0, 10, 1), e_train, label=("training set"))
    plt.plot(np.arange(0, 10, 1), e_test, label=("test set"))
    plt.legend()
    plt.show()


def exercise1_3a():
    exercise_a(s10[0], s10[1])

def exercise1_3b():
    exercise_b(s10[0], s10[1])

def exercise1_4a():    
    exercise_a(s40[0], s40[1])

def exercise1_4b():
    exercise_b(s40[0], s40[1])


def exercise1_5():
    for i in range(1, 10):
        w = PolCurFit2(s10[0], s10[1], i, np.exp(-18))
        x = np.linspace(0, 1, 100)
        y = poly(w.flatten(), x)
        plt.plot(x, y, linestyle="dashed", label=("M=" + str(i)))
    plt.plot(s10[0], s10[1], 'ro')
    plt.plot(s100[0], f(s100[0]), linewidth=1.5, label=("f"))
    plt.legend(bbox_to_anchor=(0., 0.85, 1., 0), loc=3,
           ncol=4, mode="expand", borderaxespad=0.)
    plt.show()

def exercise1_5b():
    e_train = []
    e_test = []

    rng = range(-40, 1)
    
    for lbd in rng:
        w = PolCurFit2(s10[0], s10[1], 9, np.exp(lbd))
        e_train += [error_rms(w, s10[0], s10[1])]
        e_test += [error_rms(w, s100[0], s100[1])]

    plt.plot(rng, e_train, label="Training Set")
    plt.plot(rng, e_test, label="Test Set")

    plt.ylabel("Error RMS")
    plt.xlabel("ln(lambda)")

    plt.legend()

    plt.show()

def exercise1_5c():
    polys = []
    rng = range(-20, 1)
    
    for lbd in rng:
        w = PolCurFit2(s10[0], s10[1], 9, np.exp(lbd))
        polys += [w.flatten()]

    #print(polys)
    for i in range(4,10):
        plt.plot(rng, [x[i] for x in polys], label=str(i))

    plt.legend()
    plt.show()


exercise1_5c()

'''print(np.exp(10))

for i in range(1, 10):

    w = PolCurFit2(t2, dataset, i, np.exp(-18))
    x = np.linspace(0, 0.9, 100)
    y = poly(w.flatten(), x)
    plt.plot(x, y, label=("M=" + str(i)))
    


plt.plot(t, f(t))
plt.plot(t2, dataset, 'ro')
plt.legend()
plt.show()
'''
