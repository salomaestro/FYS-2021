import numpy as np
import matplotlib.pyplot as plt
import os

dirname = os.path.dirname(__file__)

filename_blobs = os.path.join(dirname, "blobs.csv")
filename_flame = os.path.join(dirname, "flame.csv")

blobsdata = np.genfromtxt(filename_blobs, delimiter=" ")
flamedata = np.genfromtxt(filename_flame, delimiter=" ")

np.random.shuffle(blobsdata)
np.random.shuffle(flamedata)

class Cluster:
    def init(self, data):
        self.data = data

    def fit(self):
        self.prototypes = np.random.uniform(np.min(self.data), np.max(self.data), size=(3, 2))

        for vec in self.data:
            pass

    def plot(self):
        pass

def main():
    blobsplot = plt.plot(blobsdata[:, 0], blobsdata[:, 1])
    plt.show()

if __name__ == "__main__":
    main()
