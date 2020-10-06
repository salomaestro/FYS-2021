import numpy as np
import os
dirname = os.path.dirname(__file__)
filename_tictac = os.path.join(dirname, "tictac-end.csv")

"""
Pseudocode:
for j = 0, .. , d
    w_j = random(-0.01, 0.01)
repeat
    for j = 0, ... , d
        delta w_j = 0
    for t = 1, .... , N
        o = 0
        for j = 0, ... , d
            o = o + w_j * x_j^t
        y = sigmoid(o)
        for j = 0, ..., d
            delta w_j = delta w_j + (r^t - y)x_j^t
    for j=0, ... , d
        w_j = w_j + nu * delta w_j
until convergence
"""
data = np.genfromtxt(filename_tictac, delimiter=" ", dtype="int64")
groundTruth = data[:, 0]
tactoes = np.delete(data, 0, axis=1)

def gradDescent(X, r):
    w = np.random.uniform(-0.01, 0.01, size=(X.shape[1]))
    w0 = np.random.uniform(0, 10)
    s = 0.5

    for t in range(0, np.shape(X)[0]):
        dj = np.zeros_like(w)
        grddesc = True
        while grddesc:
            test = np.where(dj < 0.5, 1, 0)
            if np.all((test == 1)):
                grddesc = False
            y = 1 / (1 + np.exp(-(np.dot(w, X[t]) + w0)))
            dj = np.sum(r[t] - y) * X[t]
            print(abs(dj))
            w = w + s * dj
            w0 = w0 + s * np.sum(r[t] - y)


    c = np.dot(w.T, X) + w0
gradDescent(tactoes, groundTruth)
