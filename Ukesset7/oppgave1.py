import numpy as np
import matplotlib.pyplot as plt
import os

dirname = os.path.dirname(__file__)

filename_blobs = os.path.join(dirname, "blobs.csv")
filename_flame = os.path.join(dirname, "flame.csv")

blobsdata = np.genfromtxt(filename_blobs, delimiter=" ")
flamedata = np.genfromtxt(filename_flame, delimiter=" ")

#np.random.shuffle(blobsdata)
#np.random.shuffle(flamedata)

# Unsupervised algorithm, so no ground truth!
blobsdata = np.delete(blobsdata, 0, axis=1)
flamedata = np.delete(flamedata, 0, axis=1)

class Cluster:
    def __init__(self, data):
        self.data = data
        self.first_run = True

    def fit(self):
        np.random.seed(1)

        if self.first_run:

            # Set prototype cluster coordinate as random vectors between max and min of input data
            self.prototypes = np.random.uniform(np.min(self.data), np.max(self.data), size=(self.data.shape[1], 2))

        no_convergence = True
        runs = 0
        while no_convergence:
            runs += 1
            b = np.zeros((self.data.shape[0], self.prototypes.shape[0]))

            if not self.first_run:
                self.prototypes = np.mean(b * self.data, axis=0)
                print(self.prototypes)

            for i, vec in enumerate(self.data):

                # caluclate distances between each coordinate and possible cluster coordinate
                distances = (np.sum(self.prototypes - vec, axis=1) ** 2) ** 0.5

                # find shortest distance
                shortest = np.where(distances == np.min(distances))

                # assign this to keep track of what prototype fits best.
                b[i][shortest] = 1

            self.first_run = False

            print(runs)

    def plot(self):
        pass


def plot():
    fig, axs = plt.subplots(2, sharex=False, sharey=False)
    topplot = axs[0].scatter(blobsdata[:, 1], blobsdata[:, 0])
    bottomplot = axs[1].scatter(flamedata[:, 1], flamedata[:, 0])
    plt.show()

def main():
    blob = Cluster(blobsdata)
    blob.fit()

if __name__ == "__main__":
    main()
