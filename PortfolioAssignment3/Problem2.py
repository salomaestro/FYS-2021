import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_data
import time

frey_faces_data = load_data("frey-faces.csv")

class Cluster:
    def __init__(self, data):
        self.data = data

    def fit(self, n_clusters=2):
        # Set prototype cluster coordinate as random vectors from the original dataset, with specified amount of clusters.
        prototypesIndices = np.random.choice(len(self.data), n_clusters)
        self.prototypes = self.data[prototypesIndices]

        last_prototype = 0

        while np.sum(np.abs(self.prototypes - last_prototype)) != 0:

            last_prototype = self.prototypes

            self.b = np.zeros((self.data.shape[0], self.prototypes.shape[0]))
            self.predictions = np.zeros(self.data.shape[0])
            self.edgecases = []

            for i, vec in enumerate(self.data):

                # caluclate distances between each coordinate and possible cluster coordinate

                # using what might be a "cheatsy" way to avoid having two distances being equal, since this messes up the indices where i define shortest distance variable.. what is done is subtracting a tiny nudge of +- 1E-4 to each element of the vector.
                distances = (np.sum(self.prototypes - vec, axis=1) ** 2) ** 0.5

                # distances = distances - np.random.uniform(-1E-4, 1E-4, size=distances.shape)

                # find shortest distance
                shortest = np.where(distances == np.min(distances))[0]
                if len(shortest) > 1:
                    self.edgecases.append(i)
                    shortest = shortest[0]

                # assign this to keep track of what prototype fits best.
                self.b[i][shortest] = 1
                # print(self.predictions[i][shortest])
                self.predictions[i] = shortest


            # Calculates the mean of the datapoints assigned to a cluster. Good luck understanding this...
            cluster_mean = [np.mean(self.data[np.where(self.b[:, i] == 1)], axis=0) for i in range(self.b.shape[1])]

            self.prototypes = np.asarray(cluster_mean)
            self.predictions = np.asarray(self.predictions)

        print(self.edgecases)

        return self.prototypes, self.predictions

def rescale(vecs, original_data):
    reshapeto = (28, 20)
    if len(vecs) == 2:
        fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(10, 6))
        reference = np.random.choice(len(original_data), size=1)
        referenceimage = original_data[reference]

        labels = ["{}".format(row) for row in ["Reference\n(random images from data)", "Centroid", "Centroid"]]

        axs[0].imshow(np.reshape(referenceimage, reshapeto))

        for i, vec in enumerate(vecs, start=1):
            axs[i].imshow(np.reshape(vec, reshapeto))

        for ax, row in zip(axs, labels):
            ax.set_xlabel(row, size="large")

        fig.suptitle("Images of clusters and random reference images from the frey-face.csv file, using {} clusters".format(len(vecs)), size=14)
        fig.tight_layout()
    else:
        n = len(vecs) // 2
        fig, axs = plt.subplots(nrows=3, ncols=n, sharex=True, sharey=True, figsize=(10, 6))

        # axs[0, n // 2].set_title("Images of clusters and random reference images from the frey-face.csv file, using {} clusters".format(len(vecs)))
        fig.suptitle("Images of clusters and random reference images from the frey-face.csv file, using {} clusters".format(len(vecs)), size=14)

        reference = np.random.choice(len(original_data), size=n)
        references = original_data[reference]

        rownames = ["{}".format(row) for row in ["Row of reference\n(random images from data)", "Row of centroids", "Row of centroid"]]

        for i, vec in enumerate(references):
            axs[0, i].imshow(np.reshape(vec, reshapeto))

        for i, vec in enumerate(vecs):
            if i < n:
                axs[1, i].imshow(np.reshape(vec, reshapeto))
            else:
                i -= n
                axs[2, i].imshow(np.reshape(vec, reshapeto))

        # adds descriptive row names
        for ax, row in zip(axs, rownames):
            ax[n // 2].set_xlabel(row, size="large")
        fig.tight_layout()
    plt.show()

def main():
    start_time = time.time()
    frey = Cluster(frey_faces_data)
    frey_face, _ = frey.fit(2)

    print("Used {:.4f} seconds".format(time.time() - start_time))
    rescale(frey_face, frey_faces_data)

if __name__ == "__main__":
    main()
