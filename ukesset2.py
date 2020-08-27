import numpy as np

Pt = np.transpose(np.array(
	[[.03, .19, .27, .51], 
	[.24, .23, .33, .20], 
	[.21, .42, .19, .18], 
	[.22, .34, .08, .36]]
	))

def power_iteration(matrix, N_sim):
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

# print(power_iteration(Pt, 10))

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

print(power_iteration_v2(Pt))





