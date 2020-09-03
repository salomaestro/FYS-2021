import numpy as np
from ML_functions import power_iteration, power_iteration_v2, power_iteration_v3, PageRank, ModifiedPageRank

Pt = np.transpose(np.array(
	[[.03, .19, .27, .51], 
	[.24, .23, .33, .20], 
	[.21, .42, .19, .18], 
	[.22, .34, .08, .36]]
	))

# (a)
Amatrix = np.array(
	[[0, 0, 1, .5], 
	[1/3, 0, 0, 0], 
	[1/3, .5, 0, .5], 
	[1/3, .5, 0, 0]])
#(b)
Bmatrix = np.array(
	[[0, 1, 0, 0, 0],
	[1, 0, 0, 0, 0],
	[0, 0, 0, .5, .5],
	[0, 0, 1, 0, 0],
	[0, 0, 0, .5, .5]])

print(power_iteration(Pt, 20))
print(power_iteration_v2(Pt)[0])
print(power_iteration_v3(Pt, 20))
aNaive = PageRank(Amatrix)
aModified = ModifiedPageRank(Amatrix)
bNaive = PageRank(Bmatrix)
bModified = ModifiedPageRank(Bmatrix)
print("\nNaive PageRank method on matrix a: \n", aNaive,
	"\nModified PageRank method on matrix a: \n", aModified,
	"\nNaive PageRank method on matrix b: \n", bNaive,
	"\nModified PageRank method on matrix b: \n", bModified)

def translateResults(vec):
	"""
	Relates the ranking score to the vectors index to show which of the results are to be recommended.
	Args: 
		param: (ndarray) - must be an row vector
	returns:

	"""
	

	pass



