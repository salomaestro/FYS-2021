import numpy as np
import matplotlib.pyplot as plt
import os
from dataloader import load_data

blobsdata, flamedata = load_data("blobs.csv", "flame.csv")

np.random.shuffle(blobsdata)
np.random.shuffle(flamedata)

flamedata = flamedata
blobsdata = blobsdata

# Unsupervised algorithm, so no ground truth!
blobsdata = np.delete(blobsdata, 0, axis=1)
flamedata = np.delete(flamedata, 0, axis=1)

class Cluster:
    def __init__(self, data):
        self.data = data

    def fit(self, n_clusters=2):
        # np.random.seed(10)

        # Set prototype cluster coordinate as random vectors between max and min of input data
        self.prototypes = np.random.uniform(np.min(self.data), np.max(self.data), size=(n_clusters, 2))

        last_prototype = 0

        while np.sum(np.abs(self.prototypes - last_prototype)) != 0:

            last_prototype = self.prototypes

            b = np.zeros((self.data.shape[0], self.prototypes.shape[0]))

            for i, vec in enumerate(self.data):

                # caluclate distances between each coordinate and possible cluster coordinate
                distances = (np.sum(self.prototypes - vec, axis=1) ** 2) ** 0.5

                # find shortest distance
                shortest = np.where(distances == np.min(distances))

                # assign this to keep track of what prototype fits best.
                b[i][shortest] = 1

            # This checks wheter there is a column of b's which are assigned no datapoints. If so, recursively reduce number of clusters and fit again.
            if np.any(np.all((b == 0), axis=0)):
                print("One useless cluster, automatically reduce number of clusters by one! reduces to:", n_clusters - 1, "clusters")
                return self.fit(n_clusters - 1)

            cluster_mean = [np.mean(self.data[np.where(b[:, i] == 1)], axis=0) for i in range(b.shape[1])]

            self.prototypes = np.asarray(cluster_mean)

        return self.prototypes, n_clusters

def plot(blobclusters, flameclusters):
    fig, axs = plt.subplots(2, sharex=False, sharey=False)
    topplot = axs[0].scatter(blobsdata[:, 1], blobsdata[:, 0], label="datapoints", s=10)
    topplot = axs[0].scatter(blobclusters[:, 1], blobclusters[:, 0], label="clusters", s=50)
    axs[0].legend()
    axs[0].set_title("Blobsdata")
    bottomplot = axs[1].scatter(flamedata[:, 1], flamedata[:, 0], label="datapoints", s=10)
    bottomplot = axs[1].scatter(flameclusters[:, 1], flameclusters[:, 0], label="clusters", s=50)
    axs[1].legend()
    axs[1].set_title("Flamedata")
    fig.tight_layout()
    plt.show()

def main():
    blob = Cluster(blobsdata)
    resblobs, _ = blob.fit(2)
    flames = Cluster(flamedata)
    resflames, _ = flames.fit(10)
    # print(resflames)

    plot(resblobs, resflames)

if __name__ == "__main__":
    main()
