import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_data, load_text

inner_sweden = load_data("city-inner-sweden.csv")
names_sweden = load_text("city-names-sweden.csv")

def multidimensional_scaling(data, dimensions):
    """
    Method for doing multidimensional scaling.

    Args:
        param1: (int) - Number of dimensions to be reduced to.
    Returns:
        (ndarray) - result
    """
    B = np.matmul(data.T, data)

    # find eigenvalues and vectors
    eigenvals_n_vecs = np.linalg.eig(B)

    # sort then give indice of largest
    largest_eigenvals_indices = np.flip(np.argsort(eigenvals_n_vecs[0]))

    # find the d largest eigen indices
    wanted_eigenvals = largest_eigenvals_indices[0 : dimensions]

    # get the largest eigenvalues and vectors
    ev = eigenvals_n_vecs[0][wanted_eigenvals]
    evec = eigenvals_n_vecs[1][:, wanted_eigenvals]

    # Use the result to find Z matrix
    Lambda = np.identity(len(ev)) * ev
    Z = np.matmul(evec, Lambda ** 0.5)
    return Z

def plotcity(data, names):
    fig, axs = plt.subplots()
    axs.scatter(data[:, 0], data[:, 1])

    # add names
    for i, name in enumerate(names):
        axs.annotate(name, data[i])

    plt.ylim([-0.3e6, 0.3e6])
    plt.title("Sweden")
    plt.show()

def main():
    coordinates = multidimensional_scaling(inner_sweden, 2)
    plotcity(coordinates, names_sweden)

if __name__ == "__main__":
    main()
