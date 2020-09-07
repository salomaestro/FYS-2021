import numpy as np
from ML_functions import power_iteration, PageRank, ModifiedPageRank

Pt = np.transpose(np.array(
	[[.03, .19, .27, .51],
	[.24, .23, .33, .20],
	[.21, .42, .19, .18],
	[.22, .34, .08, .36]]
	))
# (a)
Amatrix_transposed = np.array(
	[[0, 0, 1, .5],
	[1/3, 0, 0, 0],
	[1/3, .5, 0, .5],
	[1/3, .5, 0, 0]])
#(b)
Bmatrix_transposed = np.array(
	[[0, 1, 0, 0, 0],
	[1, 0, 0, 0, 0],
	[0, 0, 0, 1, .5],
	[0, 0, 1, 0, .5],
	[0, 0, 0, 0, 0]])
# 2b
Amatrix_transposed_modified = np.array(
	[[0, 0, .5, .5, 0],
	[1/3, 0, 0, 0, 0],
	[1/3, .5, 0, .5, 1],
	[1/3, .5, 0, 0, 0],
	[0, 0, 0, .5, 0]])

print(power_iteration(Pt, 20))
aNaive = PageRank(Amatrix_transposed)
aModified = ModifiedPageRank(Amatrix_transposed)
bNaive = PageRank(Bmatrix_transposed)
bModified = ModifiedPageRank(Bmatrix_transposed)
aModified_modified  = ModifiedPageRank(Amatrix_transposed_modified)
print("\nNaive PageRank method on matrix a: \n", aNaive,
	"\nModified PageRank method on matrix a: \n", aModified,
	"\nNaive PageRank method on matrix b: \n", bNaive,
	"\nModified PageRank method on matrix b: \n", bModified,
	"\n3 added backlink to 5 and vice versa, Modified PageRank method on new matrix gives:\n", aModified_modified)

# 3 (a)
# The transition matrix is:
P_transition = np.array(
	[[.1, .25, .25],
	[.45, .5, .25],
	[.45, .25, .5]])
# 3(b)
print("Page Rank by power iteration on the transition matrix gives:\n", PageRank(P_transition))
print("Interpretation: It ranks the different weathertypes by its most probable outcome in a broader aspect.")
