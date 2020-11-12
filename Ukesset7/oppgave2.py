import numpy as np
import matplotlib.pyplot as plt
import os
from dataloader import load_data, load_text

data = load_data("city-distances.csv")
names = load_text("city-names.csv")

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
        # square proximity matrix
        SQPROXMAT = self.data ** 2

        onemat = np.ones_like(SQPROXMAT)

        # Applying double centering since data is a proximity matrix
        J = np.identity(len(SQPROXMAT)) - 1 / len(SQPROXMAT) * onemat
        B = - 0.5 * np.matmul(np.matmul(J, SQPROXMAT), J)

        # find eigenvalues and vectors
        eigenvals_n_vecs = np.linalg.eig(B)

        # sort then give indice of largest
        largest_eigenvals_indices = np.flip(np.argsort(eigenvals_n_vecs[0]))

        # find the d largest eigen indices
        wanted_eigenvals = largest_eigenvals_indices[0 : dimensions]

        # get the largest eigenvalues and vectors
        ev = eigenvals_n_vecs[0][wanted_eigenvals]
        evec = eigenvals_n_vecs[1][:, wanted_eigenvals]

        Lambda = np.identity(len(ev)) * ev
        Z = np.matmul(evec, Lambda ** 0.5)
        return Z

def plotcity(data, names):
    fig, axs = plt.subplots()
    axs.scatter(data[:, 0], data[:, 1])

    # add names
    for i, name in enumerate(names):
        axs.annotate(name, data[i])

    plt.ylim([-750, 1000])
    plt.title("Norway")
    plt.show()

def main():
    mds = FeatureReduction(data)
    res = mds.MDS(2)
    plotcity(res, names)

if __name__ == "__main__":
    main()
