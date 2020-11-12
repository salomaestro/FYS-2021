import numpy as np
import matplotlib.pyplot as plt
import os
from dataloader import load_data

blobsdata, flamedata, optdigits = load_data("blobs.csv", "flame.csv", "optdigits.csv")

np.random.shuffle(blobsdata)
np.random.shuffle(flamedata)

flamedata = flamedata
blobsdata = blobsdata

# Unsupervised algorithm, so no ground truth!
blobsdata = np.delete(blobsdata, 0, axis=1)
flamedata = np.delete(flamedata, 0, axis=1)
optdigits_labels = optdigits[:, 0]
optdigits = np.delete(optdigits, 0, axis=1)

class Cluster:
    def __init__(self, data):
        self.data = data

    def fit(self, n_clusters=2):
        # np.random.seed(11) # seed 11 was a problematic seed.

        # Set prototype cluster coordinate as random vectors from the original dataset, with specified amount of clusters.
        prototypesIndices = np.random.choice(len(self.data), n_clusters)
        self.prototypes = self.data[prototypesIndices]

        last_prototype = 0

        while np.sum(np.abs(self.prototypes - last_prototype)) != 0:

            last_prototype = self.prototypes

            self.b = np.zeros((self.data.shape[0], self.prototypes.shape[0]))
            self.predictions = np.zeros(self.data.shape[0])

            for i, vec in enumerate(self.data):

                # caluclate distances between each coordinate and possible cluster coordinate

                # using what might be a "cheatsy" way to avoid having two distances being equal, since this messes up the indices where i define shortest distance variable.. what is done is subtracting a tiny nudge of +- 1E-4 to each element of the vector.
                distances = (np.sum(self.prototypes - vec - np.random.uniform(-1E-4, 1E-4, size=self.prototypes.shape), axis=1) ** 2) ** 0.5

                # Remove this to show teacher: - np.random.uniform(-1E-4, 1E-4, size=self.prototypes.shape)

                # find shortest distance
                shortest = np.where(distances == np.min(distances))[0][0]

                # assign this to keep track of what prototype fits best.
                self.b[i][shortest] = 1
                # print(self.predictions[i][shortest])
                self.predictions[i] = shortest

            # Calculates the mean of the datapoints assigned to a cluster. Good luck understanding this...
            cluster_mean = [np.mean(self.data[np.where(self.b[:, i] == 1)], axis=0) for i in range(self.b.shape[1])]

            self.prototypes = np.asarray(cluster_mean)
            self.predictions = np.asarray(self.predictions)

        return self.prototypes, self.predictions


def plot(blobclusters, flameclusters):
    fig, axs = plt.subplots(2, sharex=False, sharey=False)
    topplot = axs[0].scatter(blobsdata[:, 0], blobsdata[:, 1], label="datapoints", s=10)
    topplot = axs[0].scatter(blobclusters[:, 0], blobclusters[:, 1], label="clusters", s=50)
    axs[0].legend()
    axs[0].set_title("Blobsdata")
    bottomplot = axs[1].scatter(flamedata[:, 0], flamedata[:, 1], label="datapoints", s=10)
    bottomplot = axs[1].scatter(flameclusters[:, 0], flameclusters[:, 1], label="clusters", s=50)
    axs[1].legend()
    axs[1].set_title("Flamedata")
    fig.tight_layout()
    plt.show()

def rescale(digs):
    n = len(digs) // 2
    fig, axs = plt.subplots(nrows=3, ncols=n, sharex=True, sharey=True, figsize=(10, 6))

    axs[0, n // 2].set_title("Images of clusters and random reference images from the optdigits file, using {} clusters".format(len(digs)))
    reference = np.random.choice(len(optdigits), size=n)
    references = optdigits[reference]

    rownames = ["{}".format(row) for row in ["Reference\n(random images)", "Centroid", "Centroid"]]

    for i, vec in enumerate(references):
        fig = axs[0, i].imshow(np.reshape(vec, (8, 8)))

    for i, vec in enumerate(digs):
        if i < n:
            fig = axs[1, i].imshow(np.reshape(vec, (8, 8)))
        else:
            i -= n
            fig = axs[2, i].imshow(np.reshape(vec, (8, 8)))

    for ax, row in zip(axs[:, 0], rownames):
        ax.set_ylabel(row, size="large")

    plt.show()

def main():
    blob = Cluster(blobsdata)
    resblobs, blobsclass = blob.fit(2)
    flames = Cluster(flamedata)
    resflames, flamesclass = flames.fit(2)

    plot(resblobs, resflames)

    opt = Cluster(optdigits)
    digits, _ = opt.fit(10)
    # print(digits, _)
    rescale(digits)


if __name__ == "__main__":
    main()
