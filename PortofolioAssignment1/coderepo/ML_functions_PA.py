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

	a = np.zeros(np.shape(matrix)[0])
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
	return PageRank(G.T)

def main():
	return None

if __name__ == "__name__":
	main()
