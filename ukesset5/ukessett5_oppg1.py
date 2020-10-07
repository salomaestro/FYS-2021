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
# Import and initialize the data
data = np.genfromtxt(filename_tictac, delimiter=" ", dtype="int64")
groundTruth = data[:, 0]
tactoes = np.delete(data, 0, axis=1)

def gradDescent(X, r):
    # Define random vector of weights with the same dimension as our data, 9 in this case
    w = np.random.uniform(-0.01, 0.01, size=(X.shape[1]))

    # Define random bias
    w0 = 100

    # Step size, may change later.
    s = 5

    i = np.zeros(np.shape(X)[0])
    # Iterate throught each result (row of data)
    for t in range(0, np.shape(X)[0]):
        # Create array of ones with same shape as the weights this is the derivatives
        Dj = np.ones_like(w) * 10
        grddesc = True
        while grddesc:
            # Sigmoid
            y = 1 / (1 + np.exp(-(np.dot(w.T, X[t]) + w0)))
            Dj = (r[t] - y) * X[t]
            w = w + s * Dj
            w0 = w0 + s * r[t] - y
            test = np.where(Dj < 0.05, 1, 0)
            if np.all((test == 1)):
                grddesc = False
            i[t] += 1

    c = np.dot(w, X.T) + w0
    print(c[c < 0].shape)
gradDescent(tactoes, groundTruth)
