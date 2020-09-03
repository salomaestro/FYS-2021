import numpy as np

# 1
# print(np.__version__, "\n", np.show_config())
# 2
# print(np.info(np.add))
# 3 sjekke om ingen elementer er null
arr = np.arange(0, 11)
# print(np.all(arr))
# 4 sjekke om alle null
# print(np.any(arr))
# 5 sjekke om noen elementer er nan eller inf
infarr = np.array([1, 0, np.nan, np.inf])
# print(np.isfinite(infarr))
# 6 sjekke pos/neg inf
# print(np.isinf(infarr))
# 7 nan
# print(np.isnan(infarr))
# 8 teste compleks/reell plus scalar eller ikke
comparr = np.array([2+1j, 1+4j, 4.6, 3, 2, 2j])
# print(np.iscomplex(comparr))
# print(np.isreal(comparr))
# print(np.isscalar(4.6), np.isscalar([4.6]))
# 9 sjekke om to arrays er like med feilmargin
a1 = np.array([15, 1, 26, 100])
a2 = np.array([15, 1, 26, 100.001])
# print(np.equal(a1, a2))
# print(np.allclose(a1, a2))
# 10
b1 = np.array([1, 4])
b2 = np.array([2, 4])
# print(np.greater(b1, b2))
# print(np.greater_equal(b1, b2))
# print(np.less(b1, b2))
# 12
somearr = np.array([1, 7, 13, 105])
# print("%d bytes" % (somearr.size * somearr.itemsize))
# 13
# print(np.zeros(10), np.ones(10), np.ones(10)*5)
# 14
# print(np.arange(30, 71))
# 15
# print(np.arange(30, 71, 2))
# 16
# print(np.identity(3))
# 17
# print(np.random.uniform(0, 1))
# 18
# print(np.random.normal(0, 1, 15))
# 19
# print(np.arange(15, 56)[1:-1])
# 20
rand_arr = np.arange(10, 22).reshape(3, 4)
# print(rand_arr)
# for x in np.nditer(rand_arr):
#     print(x, end=" ")
# 21
# print(np.linspace(5, 50, 10))
# 22
vec = np.arange(21)
# print(vec)
vec[(vec >= 9) & (vec <= 15)] *= -1
# print(vec)
# 23 tilfeldige tall mellom 0 og 10 med lengde 5
# print(np.random.randint(0, 11, 5))
# 24
vec1 = np.random.randint(0, 2, 3)
vec2 = np.random.randint(0, 2, 3)
# print(vec1 * vec2)
# 25
Matrix = np.arange(10, 22).reshape(3, 4)
print(Matrix)
# 26
def nrowsandcols(mat):
	"""
	Find the number of rows and columns of given matrix
	Args:
		param: (ndarray) - matrix
	Returns:
		(int, int) - rows, cols
	"""
	return len(mat), len(mat[0])

# print(nrowsandcols(Matrix))
# 27
# print(np.eye(3))
# 28
onematrix = np.ones((10, 10))
onematrix[1:-1, 1:-1] = 0
# print(onematrix)
# 29
diagonalMatrix = np.diag([1, 2, 3, 4, 5])
# print(diagonalMatrix)
# 30
staggMatrix = np.zeros([4, 4])
staggMatrix[::2, 1::2] = 1
staggMatrix[1::2, ::2] = 1
print(staggMatrix)








