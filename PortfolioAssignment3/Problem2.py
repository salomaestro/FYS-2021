import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_data
import time

def get_data():
    data = load_data("frey-faces.csv")
    return data

def K_means(data, n=2):
    """
    Function which uses the K-means algorithm to find clusters unsupervised in a dataset.

    Args:
        param1: (np.ndarray) - The data which is to be used.
        param2: (int) - optional, standard set to 2. specifies how many clusters.

    Returns:
        Prototypes/clusters - (np.ndarray). n-dimensional coordinates,
                    which specify the position of the clusters
        Predicitons - (np.ndarray). Specifies which unique cluster a
                    atapoint belongs to.
        Bordercases - (np.ndarray). The indices of the images which lay
                    close to or on the border between two or more clusters.
        Closest - (np.ndarray). The indices of the images which lay
                    closest to each cluster.
    """

    # Set prototype cluster coordinate as random vectors from the original dataset, with specified amount of clusters.
    prototypesIndices = np.random.choice(len(data), n)
    prototypes = data[prototypesIndices]

    last_prototype = 0

    while np.sum(np.abs(prototypes - last_prototype)) != 0:

        last_prototype = prototypes

        # Will fill this array with the shortest distance calculated.
        closest_to_cluster = np.ones((data.shape[0], prototypes.shape[0])) * np.inf

        # Indices of the edgecases
        edgecases = []

        # specifies which cluster each datapoint are closest to.
        b = np.zeros((data.shape[0], prototypes.shape[0]))
        predictions = np.zeros(data.shape[0])

        for i, vec in enumerate(data):

            # caluclate distances between each coordinate and possible cluster coordinate.
            distances = (np.sum(prototypes - vec, axis=1) ** 2) ** 0.5

            # Finds the images that lay on the border between two clusters. i have used a threshold to check this of 7, such that if the difference in distance between two vectors are less than seven, declare border-incident.
            border = np.argsort(distances)
            if abs(distances[border[0]] - distances[border[1]]) < 7:
                edgecases.append(i)

            # find shortest distance
            shortest = border[0]

            # assign this to keep track of what prototype fits best.
            b[i][shortest] = 1
            predictions[i] = shortest
            closest_to_cluster[i][shortest] = distances[shortest]

        # Calculates the mean of the datapoints assigned to a cluster, along the column axis.
        cluster_mean = [np.mean(data[np.where(b[:, i] == 1)], axis=0) for i in range(b.shape[1])]

        prototypes = np.asarray(cluster_mean)
        predictions = np.asarray(predictions)
        closest_to_each_cluster = np.argsort(closest_to_cluster, axis=0)[0]

    edgecases = np.asarray(edgecases)

    return prototypes, predictions, edgecases, closest_to_each_cluster

def rescale(vecs, original_data, edgecase, closest):
    # Settings for plotting.
    figsize = (10, 6)
    newshape = (28, 20)
    cmap = "gray"
    n = len(vecs) // 2

    # plot!
    fig, axs = plt.subplots(nrows=4, ncols=n, sharex=True, sharey=True, figsize=figsize)

    # Include names at rows
    rownames = ["{}".format(row) for row in ["Centroids", "Closest to the centroids above", "Centroids", "Closest to the centroids above"]]

    # Tailor the plotting for the 2 cluster case and for the 4 and 10 cluster so we can have them somewhat nice.
    if n > 1:
        for ax, name in zip(axs, rownames):
            ax[n // 2].set_xlabel(name, size="large")

        for i, vec in enumerate(zip(vecs, closest)):
            if i < n:
                axs[0, i].imshow(np.reshape(vec[0], newshape), cmap=cmap)
                axs[1, i].imshow(np.reshape(original_data[vec[1]], newshape), cmap=cmap)
            else:
                i -= n
                axs[2, i].imshow(np.reshape(vec[0], newshape), cmap=cmap)
                axs[3, i].imshow(np.reshape(original_data[vec[1]], newshape), cmap=cmap)
    else:
        for ax, name in zip(axs, rownames):
            ax.set_xlabel(name, size="large")

        for i, vec in enumerate(zip(vecs, closest)):
            if i < n:
                axs[0].imshow(np.reshape(vec[0], newshape), cmap=cmap)
                axs[1].imshow(np.reshape(original_data[vec[1]], newshape), cmap=cmap)
            else:
                i -= n
                axs[2].imshow(np.reshape(vec[0], newshape), cmap=cmap)
                axs[3].imshow(np.reshape(original_data[vec[1]], newshape), cmap=cmap)

        for i, ax in enumerate(axs):
            if i % 2 == 0:
                ax.imshow(np.reshape(vecs[i // 2], newshape), cmap=cmap)
            else:
                ax.imshow(np.reshape(original_data[closest[i // 2]], newshape), cmap=cmap)

    # Add title, make it tight and remove the axes...
    fig.suptitle("Images of clusters and their closest image corresponding to cluster, using {} clusters".format(len(vecs)), size=14)
    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Plotting the border cases!
    ednum = len(edgecase)
    if ednum != 0:
        if ednum > 5:
            # To not make the plot too large i at max only include 5 images.
            edgeind = np.random.choice(len(edgecase), size=5)
            edgecase = edgecase[edgeind]
        fig, axs = plt.subplots(ncols=len(edgecase), figsize=figsize)

        if ednum != 1:
            for edge, ax in zip(edgecase, axs):
                ax.imshow(np.reshape(original_data[edge], newshape), cmap=cmap)
                ax.set_axis_off()
        else:
            # For the event where there is only one border incident.
            axs.imshow(np.reshape(original_data[edgecase], newshape), cmap=cmap)
            axs.set_axis_off()

        fig.suptitle("For {} clusters there are {} bordercase(s).".format(len(vecs), ednum), size=18)
        fig.tight_layout()
        plt.show()

def main():
    # To recreate the results i presented in my paper use the seed 69420.
    # np.random.seed(69420)
    frey_faces_data = get_data()

    K = [2, 4, 10]

    for k in K:
        cluster, prediction, edgecase, closest = K_means(frey_faces_data, k)
        rescale(cluster, frey_faces_data, edgecase, closest)

if __name__ == "__main__":
    main()
