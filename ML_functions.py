import numpy as np


def power_iteration_v3(matrix, N_sim):
	"""
	Iterates like the power iteration method to converge on the stationary probabilities of the transitionmatrix

	Args:
		param1: (ndarray) - Transition matrix
		parma2: (int) - Number of iterations
	Returns:
		(ndarray) - Stationary probability matrix
	"""
	la = np.random.rand(matrix.shape[1])

	for _ in range(N_sim):
	
		la1 = np.dot(matrix, la)
		
		la1_norm = np.linalg.norm(la1, 1)
		
		la = la1 / la1_norm

	return la

def power_iteration_v2(matrix):
	"""
	Iterates like the power iteration method to converge on the stationary probabilities of the transitionmatrix,
	but when the difference is small enough 

	Args:
		param1: (ndarray) - Transition matrix
	Returns:
		(ndarray) - Stationary probability matrix
		(float) - eigenvalue
		(int) - number of iteratios
	"""
	# creates a vector of random float values as a row vector with the same size of rows as given matrix
	v = np.random.rand(matrix.shape[1])

	# finds eigenvalues
	ev = matrix.dot(v)
	ev = v.dot(ev)

	iterations = 0

	while True:
		Av = matrix.dot(v)
		v_new = Av / np.linalg.norm(Av, 1)

		ev_new = matrix.dot(v_new)
		ev_new = v_new.dot(ev_new)

		if np.abs(ev - ev_new) < 1E-15:
			break

		v = v_new
		ev = ev_new
		iterations += 1

	return v_new, ev_new, iterations

def power_iteration(matrix, i):
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
	temp = ev.argsort()
	ranks = np.empty_like(temp)
	ranks[temp] = np.arange(len(ev))[::-1]
	return ranks

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
	b = np.ones(np.shape(H)[0]) / (np.shape(H)[0] + 1)
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