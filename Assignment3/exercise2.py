import numpy as np

def irls_sin(x_0, count):
	x_1 = x_0 + np.cos(x_0) / np.sin(x_0)
	return [x_1] if count <= 1 else [x_1] + irls_sin(x_1, count-1)

print(irls_sin(1, 5))
print(irls_sin(-1, 5))

phi_data = np.array([0.3, 0.44, 0.46, 0.6])
phi = np.array([[1, p] for p in phi_data])

t_data = np.array([1.0, 0, 1.0, 0])
w_0 = np.array([1.0, 1.0])


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def gE(w):
	y = sigmoid(np.dot(w.T, phi.T))
	return np.dot(phi.T, (y - t_data))


def H(w):
	ys =sigmoid(np.dot(w.T, phi.T))
	R = np.diag([ (y*(1-y)) for y in ys])

	return np.dot(np.dot(phi.T, R),phi)


def weight(w):
	return w - np.dot(np.linalg.inv(H(w)), gE(w))

def irls_cee(w_0, count):
	w_1 = w_0 - np.dot(np.linalg.inv(H(w_0)), gE(w_0))
	return [w_1] if count <= 1 else [w_1] + irls_cee(w_1, count-1)

print(irls_cee(w_0, 10))



