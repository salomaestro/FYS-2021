import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# Problem 3 (a)
data = np.genfromtxt("global-temperatures.csv", delimiter=" ")

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
        self.regressionline = self.w1 * self.x + self.w0
        return (self.x, self.regressionline)

    def r_squared(self):
        """
        Calculates SST, SSE, SSR, R^2
        """
        r_hat = np.matmul(self.X, self.w)
        SSE = np.sum((self.r - r_hat) ** 2)
        SST = np.sum((self.r - np.mean(self.r)) ** 2)
        R_squared = 1 - SSE/SST
        return R_squared

    def residuals(self):
        """
        Calculates the residuals at each year, by taking the difference between the
        regression line and the actual value
        Args:
            param1: (self)
        Returns:
            (ndarray) - contains floats of the residuals.
        """
        residual = np.abs(self.regressionline - self.y)
        return residual

reg = LinearRegression(data)
result = reg.least_squares()

#### scipy comparison
slope, intercept, r_value, p_value, std_err = stats.linregress(data)
y = slope * data[1] + intercept
####
# Problem (3b)
# The R^2 value tells us how good the regression line fits to the given data.
# it´s calculated as R^2 = SSR/SST =(SST - SSE)/SST = 1 - SSE/SST
print("R^2 value: ", reg.r_squared())
# Problem (3c)
# we can interprit the estimator betta^1 as the slope of our regression line
# Problem (3d)
# plotting the resiudals means plotting the error of our regression analysis.
residual_error = reg.residuals()

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
plt.hist(residual_error)
plt.xlabel("Error")
plt.ylabel("Number of Points")
plt.legend()
plt.show()