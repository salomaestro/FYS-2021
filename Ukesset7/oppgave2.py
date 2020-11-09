import numpy as np
import os
from dataloader import load_data

data = load_data("city-distances.csv")

class FeatureReduction:
    def __init__(self, data):
        self.data = data

    def MDS(self, dimensions):
        """
        Method for doing multidimensional scaling.

        Args:
            param1: (int) - Number of dimensions to be reduced to.
        Returns:
            (ndarray) - result
        """
        X = self.data
        eigenvals_n_vecs = np.linalg.eig(np.matmul(X.T, X))
        D = eigenvals_n_vecs[0]
        D = np.identity(len(D)) * D
        




def main():
    mds = FeatureReduction(data)
    res = mds.MDS(2)

if __name__ == "__main__":
    main()
