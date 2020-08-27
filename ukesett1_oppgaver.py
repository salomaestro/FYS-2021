import numpy as np

# A = np.random.randint(0, 6, (4, 4))
# B = np.random.randint(0, 6, (4, 4))
# print("A = \n{}\nB = \n{}".format(A, B))
n = 4
# OPPGAVE 1B ############
a = np.random.randint(0, 6, (n, 1))
b = np.random.randint(0, 6, (n, 1))

print("a = {}\nb = {}".format(np.transpose(a), np.transpose(b)))

def dotColumnVectors(vec1, vec2):
	"""
	Args:
		param1: (np.ndarray) - first vector
		param2: (np.ndarray) - second vector
	Returns:
		(int) - dot product
	"""
	product = np.dot(vec1.T, vec2)
	return product[0][0]

dotProduct = dotColumnVectors(a, b)
print("\na . b = {}".format(dotProduct))

# OPPGAVE 1C

def summingLoop(vec1, vec2):
	"""
	Args:
		param1: (np.ndarray) - first vector
		param2: (np.ndarray) - second vector
	Returns:
		(int) - sum of elements (dot product?)
	"""
	totalSum = dotColumnVectors(vec1, vec2)
	return totalSum

print("Two ways of writing the same thing..\n => a . b = {}".format(summingLoop(a, b)))

# Oppgave 1D
# random matrix:
X = np.random.randint(0, 6, (5, n))
print(X)
Y = np.dot(X, a)

print("X * a = \n{}".format(Y))

Ysummed = sum(np.dot(X, a))
print(Ysummed[0])