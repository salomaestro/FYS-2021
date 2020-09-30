import numpy as np
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
