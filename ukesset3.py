import numpy as np
import matplotlib.pyplot as plt
import csv
# from sklearn.linear_model import LinearRegression as LR

# Problem 3 (a)
data = np.genfromtxt("FYS-2021\global-temperatures.csv", delimiter=" ")

class LinearRegression:
    """
    Should have the shape of an (N x 2) matrix or (2 x N) matrix transposed
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.x = self.dataset.T[0,:]
        self.y = self.dataset.T[1,:]
        self.N = np.shape(self.dataset)[0]
        self.A = np.array(
            [[self.N, np.sum(self.x)],
            [np.sum(self.x), np.sum(self.x ** 2)]])
        self.X = [np.ones(self.N).T]
        self.X = np.concatenate((self.X, [self.x])).T
        self.r = np.array([self.y]).T


    def least_squares(self):
        """
        Linear regression by least squares method by linear algebra method
        Args:
            (self) - instance
        Returns:
            (ndarray, ndarray) - (x, y) ready for plotting
        """
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T), self.r)
        self.w0 = self.w[0][0]
        self.w1 = self.w[1][0]
        y = self.w1 * self.x + self.w0
        return (self.x, y)

reg = LinearRegression(data)
result = reg.least_squares()

#### scikit part

####

plt.plot(reg.x, reg.y, "g.")
plt.plot(result[0], result[1])
plt.xlabel("years")
plt.ylabel("Temperature")
plt.title("Global temperatures")
plt.legend()
plt.show()
