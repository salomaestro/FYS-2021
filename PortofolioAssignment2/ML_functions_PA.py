import numpy as np
import math

# Candidate 25

def power_iteration(matrix, i):
	"""
	Iterates like the power iteration method to converge on the stationary probabilities of the transitionmatrix
	Args:
		param1: (ndarray) - Transition matrix
		parma2: (int) - Number of iterations
	Returns:
		(ndarray) - Stationary probability matrix
	"""
	# Our matrix has to be an transition matrix, also called a probability matrix, which implies it has to be stochastic, meaning the columns must sum to one.
	if np.sum(matrix, axis=1)[0] == 1:
		matrix = matrix.T
	if np.sum(matrix, axis=0)[0] != 1:
		print("This is not a stochastic/probability matrix")

	# Random eigenvector
	pi = np.random.rand(matrix.shape[1])

	# Normalizing
	pi_norm = pi / np.linalg.norm(pi, 1)

	# Power iterating
	pow_matrix = np.matrix(matrix) ** i

	# Return the product
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
	# Using power_iteration to create eigenvector
	ev = power_iteration(matrix, 100)

	# Cast to correct shape
	ev = np.array(ev)[0]

	# Sort by indices
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

		self.X = np.concatenate((self.X, self.x.reshape(1, self.N))).T
		self.r = np.array(self.y).T

	def estimators(self):
		"""
		Calculate the estimator vector w by the method w* = (X^T X)^-1 X^T r
		Returns:
			(numpy.matrix) - A column vector where the first element is the y-intercept, and the
							 secound is the slope of the regression function.
		"""
		# Formulae for calculating the estimators
		self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), self.X.T), self.r)

		# Extract estimators
		self.w0 = self.w[0]
		self.w1 = self.w[1]
		return self.w0, self.w1

	def least_squares(self):
		"""
		Linear regression by least squares method by linear algebra method
		Returns:
			(ndarray, ndarray) - (x, y) ready for plotting
		"""
		# Make sure estimators have been created
		self.estimators()

		# Create array consisting of values for regression line
		self.regressionline = self.w1 * self.x + self.w0
		return (self.x, self.regressionline)

	def meanSquaredError(self):
		"""
		Calculates the mean squared error
		Returns:
			(float) - MSE
		"""
		# Estimator
		r_hat = np.array(np.matmul(self.X, self.w))

		# Generic mean squared error
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
		Returns:
		(ndarray) - contains floats of the residuals.
		"""
		self.least_squares()
		residual = np.abs(self.regressionline - self.y)
		return residual

class ProbabilityDistributions:
	"""
	Class used for implementing the probability distributions needed in Portofolio assignment 1
	"""
	def __init__(self, data):
		self.x = data

	def gamma(self, alpha, betta):
		"""
		Generic function for calculating the Gaussian distibution.

		Returns:
			(ndarray, ndarray) - (x, y)
		"""
		dist = 1 / (betta ** alpha * math.gamma(alpha)) * self.x ** (alpha - 1) * np.exp(- self.x / betta)
		return self.x, dist

	def normal(self, mean, var):
		"""
		Generic function for calculating the normal (or gaussian) distribution.
		Returns:
		(ndarray, ndarray) - (x, y)
		"""
		dist = 1 / (np.sqrt(2 * np.pi) * np.sqrt(var)) * np.exp(- (self.x - mean) ** 2 / (2 * var))
		return self.x, dist

# Creating the bayes classifier
def Bayes_classifier(all_x, priorclass0, priorclass1, alpha, betta_hat, mean, var):
	"""
	Method for classifying handwritten 0's and 1's under the assumption that all 0's follow a Gamma distribution, and all 1's follow a Normal distribution.
	Args:
	    param1: (ndarray) - Collection of 0's and 1's interpreted by a camera as floats i, where 0 <= i <= 1.
	    param2: (float) - prior probability of belonging to class 0.
	    param3: (float) - prior probability of belonging to class 1.
		peram4: (float) - defined alpha value for the gamma distribution
		param5: (float) - estimated betta value for the gamma distibution
		param6: (float) - estimated mean
		param7: (float) - estimated variance
	Returns:
	    output1: (ndarray) - First row: index of values classified as 0's, second row: values of classified 0's.
	    output2: (ndarray) - First row: index of values classified as 1's, second row: values of classified 1's.
	"""
	# Uses my previously created class containing methods for calculating normal and gamma distributions
	classify = ProbabilityDistributions(all_x)

	# p(x|C_i) - using our distributions
	likelihood0 = classify.gamma(alpha, betta_hat)[1]
	likelihood1 = classify.normal(mean, var)[1]

	# Classifying by index
	C1_index = np.where(likelihood1 * priorclass1 > likelihood0 * priorclass0)
	C0_index = np.where(likelihood0 * priorclass0 > likelihood1 * priorclass1)

	# Assigning the classified index to classes
	C1 = all_x[C1_index]
	C0 = all_x[C0_index]

	# Reshaping reshaping such that we can concatenate
	C0 = C0.reshape(1, C0.shape[0])
	C1 = C1.reshape(1, C1.shape[0])

	# Doing the same for the indices
	C0_index = C0_index[0].reshape(1, C0.shape[1])
	C1_index = C1_index[0].reshape(1, C1.shape[1])

	# Concatenating both
	C0_withindex = np.concatenate((C0_index, C0))
	C1_withindex = np.concatenate((C1_index, C1))
	return C0_withindex, C1_withindex

def main():
	return None

if __name__ == "__name__":
	main()
