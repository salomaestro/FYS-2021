import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_data
import time

frey_faces_data = load_data("frey-faces.csv")

def K_means(data, n_clusters=2):
    
    # Set prototype cluster coordinate as random vectors from the original dataset, with specified amount of clusters.
    prototypesIndices = np.random.choice(len(data), n_clusters)
    prototypes = data[prototypesIndices]

    last_prototype = 0

    while np.sum(np.abs(prototypes - last_prototype)) != 0:

        last_prototype = prototypes

        closest_to_cluster = np.ones((data.shape[0], prototypes.shape[0])) * np.inf
        edgecases = []

        b = np.zeros((data.shape[0], prototypes.shape[0]))
        predictions = np.zeros(data.shape[0])

        for i, vec in enumerate(data):

            # caluclate distances between each coordinate and possible cluster coordinate.
            distances = (np.sum(prototypes - vec, axis=1) ** 2) ** 0.5

            # Finds the images that lay on the border between two clusters. i have used a threshold to check this of 2, such that if the difference in distance between two vectors are less than two, declare border-incident.
            for dist in distances:
                # elementwise compare each distance to list of distances. There will always be at least one which is true, therefore check when the length is greater than 1.
                if len(distances[np.abs(distances - dist) < 2]) > 1:
                    edgecases.append(i)


            # find shortest distance
            shortest = np.argsort(distances)[0]

            # assign this to keep track of what prototype fits best.
            b[i][shortest] = 1
            predictions[i] = shortest
            closest_to_cluster[i][shortest] = distances[shortest]

        # Calculates the mean of the datapoints assigneborder casesd to a cluster. Good luck understanding this...
        cluster_mean = [np.mean(data[np.where(b[:, i] == 1)], axis=0) for i in range(b.shape[1])]

        prototypes = np.asarray(cluster_mean)
        predictions = np.asarray(predictions)
        closest_to_each_cluster = np.argsort(closest_to_cluster, axis=0)[0]

    edgecases = np.unique(np.asarray(edgecases))
    print(edgecases)
    return prototypes, predictions, edgecases, closest_to_each_cluster

def rescale(vecs, original_data, edgecase, closest):
    # Shape which we wish to reshape array to.
    figsize = (10, 6)
    newshape = (28, 20)
    n = len(vecs) // 2
    fig, axs = plt.subplots(nrows=4, ncols=n, sharex=True, sharey=True, figsize=figsize)

    rownames = ["{}".format(row) for row in ["Centroids", "Closest to the centroids above", "Centroids", "Closest to the centroids above"]]
    if n > 1:
        for ax, name in zip(axs, rownames):
            ax[n // 2].set_xlabel(name, size="large")

        for i, vec in enumerate(zip(vecs, closest)):
            if i < n:
                axs[0, i].imshow(np.reshape(vec[0], newshape))
                axs[1, i].imshow(np.reshape(original_data[vec[1]], newshape))
            else:
                i -= n
                axs[2, i].imshow(np.reshape(vec[0], newshape))
                axs[3, i].imshow(np.reshape(original_data[vec[1]], newshape))
    else:
        for ax, name in zip(axs, rownames):
            ax.set_xlabel(name, size="large")

        for i, vec in enumerate(zip(vecs, closest)):
            if i < n:
                axs[0].imshow(np.reshape(vec[0], newshape))
                axs[1].imshow(np.reshape(original_data[vec[1]], newshape))
            else:
                i -= n
                axs[2].imshow(np.reshape(vec[0], newshape))
                axs[3].imshow(np.reshape(original_data[vec[1]], newshape))

        for i, ax in enumerate(axs):
            if i % 2 == 0:
                ax.imshow(np.reshape(vecs[i // 2], newshape))
            else:
                ax.imshow(np.reshape(original_data[closest[i // 2]], newshape))

    fig.suptitle("Images of clusters and their closest image corresponding to cluster, using {} clusters".format(len(vecs)), size=14)
    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()

    if len(edgecase) != 0:
        if len(edgecase) > 1:
            if len(edgecase) > 8:
                # if we have more than 6, choose six random.
                edgecaseind = np.random.choice(len(edgecase), 8)
                edgecase = edgecase[edgecaseind]
            n = len(edgecase) // 2
            fig, axs = plt.subplots(nrows=2, ncols = n, figsize=figsize)

            for i, edge in enumerate(edgecase):
                if i < n:
                    axs[0, i].imshow(np.reshape(original_data[edge], newshape))
                else:
                    i -= n
                    axs[1, i].imshow(np.reshape(original_data[edge], newshape))

            fig.suptitle("For {} clusters, we have the border cases where\ntheese images lay on the border between two clusters.".format(len(vecs)), size=14)
        else:
            fig, ax = plt.subplots(1, figsize=figsize)
            ax.imshow(np.reshape(original_data[edgecase], newshape))
            fig.suptitle("For {} clusters, we have the border cases where\nthis image lay on the border between two clusters.".format(len(vecs)), size=14)

        fig.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.show()

def main():
    start_time = time.time()

    K = [2, 4, 10]

    for k in K:
        pass

    two = K_means(frey_faces_data, 2)
    four = K_means(frey_faces_data, 4)
    tenner = K_means(frey_faces_data, 10)

    print("Used {:.4f} seconds".format(time.time() - start_time))

    rescale(two[0], frey_faces_data, two[2], two[3])
    rescale(four[0], frey_faces_data, four[2], four[3])
    rescale(tenner[0], frey_faces_data, tenner[2], tenner[3])

if __name__ == "__main__":
    main()
