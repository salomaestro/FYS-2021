import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# Problem 3 (a)
data = np.genfromtxt("FYS-2021\global-temperatures.csv", delimiter=" ")

class LinearRegression:
    """
    Should have the shape of an (N x 2) matrix or (2 x N) matrix transposed
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.x = self.dataset.T[0, :]
        self.y = self.dataset.T[1, :]
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

#### scipy comparison
<<<<<<< HEAD
slope, intercept, r_value, p_value, std_err = stats.linregress(data)
y = slope * data[1] + intercept
=======
# slope, intercept, r_value, p_value, std_err = stats.linregress(data.T[0, :], data.T[1, :])
>>>>>>> 2e1c32c18423e5948d4a01a966ea47be9d3794c1
####

plt.plot(reg.x, reg.y, ".")
plt.plot(data[1], y, "y")
plt.plot(result[0], result[1], "g")
plt.xlim(data.T[0][0], data.T[0][-1])
plt.ylim(np.min(data.T[1,:]), np.max(data.T[1,:]))
plt.xlabel("years")
plt.ylabel("Temperature")
plt.title("Global temperatures with least squares linear regression")
plt.legend()
plt.show()
