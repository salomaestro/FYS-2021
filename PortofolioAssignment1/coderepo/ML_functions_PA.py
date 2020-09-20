import numpy as np

def power_iteration(matrix, i):
	"""
	Iterates like the power iteration method to converge on the stationary probabilities of the transitionmatrix

	Args:
		param1: (ndarray) - Transition matrix
		parma2: (int) - Number of iterations
	Returns:
		(ndarray) - Stationary probability matrix
	"""
	pi = np.random.rand(matrix.shape[1])
	pi_norm = pi / np.linalg.norm(pi, 1)

	pow_matrix = np.matrix(matrix) ** i
	return pow_matrix.dot(pi_norm)

def PageRank(matrix):
	"""
	Uses the PageRank method to list most relevant suggestions ordered
	The ordering goes from 0 which is the best result, to n which is the worst result.

	Args:
		(numpy.matrix) - Transitionmatrix
	Returns:
		(ndarray) - ranked
	"""
	ev = power_iteration(matrix, 100)
	ev = np.array(ev)[0]
	rank = np.argsort(ev)
	return rank

def ModifiedPageRank(matrix, alpha=0.85):
	"""
	Googles implementation of the PageRank method, Uses the principle G = alpha * S + (1 - alpha) * E
	The ordering goes from 0 which is the best result, to n which is the worst result

	Args:
		param1: (ndarray) - matrix, should be of size n x n
		param2: (float) - optional, 0 < alpha < 1, determines how zeros and ones should be converted to floats, follows links, else, URL.

	Returns:
		(ndarray) - list of ranks
	"""
	# Should implement a system for not using the first row if all elements are zero.
	# if H[0,:].all():
	# 	pass

	a = np.zeros(np.shape(matrix)[1])
	a[0] = 1 / (np.count_nonzero(matrix.T[0,:]) + 1)


	# Concatenates a new row vector to the bottom of the original matrix
	H = np.concatenate((matrix, [a]))
	firstRow = H.T[0,:]
	firstRow[firstRow > 0] = a[0]

	# Concatenates a new row with 1/N to the bottom of the transposed matrix H
	b = np.ones(np.shape(H)[0]) / (np.shape(H)[0])
	S = np.concatenate((H.T, [b]))

	E = np.ones(np.shape(S)) / np.shape(S)[0]

	# G is the Google matrix
	G = alpha * S + (1 - alpha) * E

	# Now the ranking begins
	return G.T

class LinearRegression:
	"""
	Class for using linear regression on datasets.
	Should have the shape of an (N x 2) matrix or (2 x N) matrix transposed
	"""
	def __init__(self, dataset):
		self.dataset = dataset

		# Splits the dataset into two ndarrays
		self.x = self.dataset.T[0, :]
		self.y = self.dataset.T[1, :]

		# Shape of the dataset
		self.N = np.shape(self.dataset)[0]

		# Create X matrix of ones, then concatenate X with x (from dataset) to give the wanted X matrix
		self.X = np.ones(self.N)
		self.X = self.X.reshape((1, self.N))
		self.X = np.concatenate((self.X, self.x)).T
		self.r = np.array(self.y).T

	def estimators(self):
		"""
		Calculate the estimator vector w by the method w* = (X^T X)^-1 X^T r
		Returns:
			(numpy.matrix) - A column vector where the first element is the y-intercept, and the
							 secound is the slope of the regression function.
		"""
		self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T), self.r)
		self.w0 = self.w[0][0].item()
		self.w1 = self.w[1][0].item()
		return self.w0, self.w1

	def least_squares(self):
		"""
		Linear regression by least squares method by linear algebra method
		Returns:
			(ndarray, ndarray) - (x, y) ready for plotting
		"""
		self.estimators()
		self.regressionline = self.w1 * self.x + self.w0
		return (self.x, self.regressionline)

	def meanSquaredError(self):
		"""
		Calculates the mean squared error
		Returns:
			(float) - MSE
		"""
		r_hat = np.array(np.matmul(self.X, self.w))

		self.MSE = np.sum((np.array(self.r) - r_hat) ** 2)
		return self.MSE

	def totalSumSquares(self):
		"""
		Calculates the total sum of squares
		Returns:
			(float) - TSS
		"""
		self.TSS = np.sum((self.r - np.mean(self.r)) ** 2)
		return self.TSS

	def r_squared(self):
		"""
		Calculates R^2
		Returns:
			(float) - R^2 value
		"""
		self.meanSquaredError()
		self.totalSumSquares()
		R_squared = 1 - self.MSE / self.TSS
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

def main():
	return None

if __name__ == "__name__":
	main()
