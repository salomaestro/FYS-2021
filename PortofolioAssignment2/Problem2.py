import numpy as np
import matplotlib.pyplot as plt
import os
# from sklearn import metrics

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
filename_seals_test = os.path.join(dirname, "seals_test.csv")
filename_seals_images_test = os.path.join(dirname, "seals_images_test.csv")
traindata = np.genfromtxt(filename_seals_train, delimiter=" ")
testdata = np.genfromtxt(filename_seals_test, delimiter=" ")

ground_truth = traindata[:, 0]
traindata = np.delete(traindata, 0, axis=1)

class Tree:
    """
    Class for decision tree.
    """
    def __init__(self, trainingData, gt):
        # Extract ground truth by indices
        self.gt = gt

        # Init training data
        self.traindata = trainingData

        self.minimum_impurity = 1

    def fit(self, data, gt):
        if self.impurity(gt) < self.minimum_impurity:
            self.minimum_impurity = self.impurity(gt)
            pass
        else:
            # Find best split
            split_index, threshold = self.find_best_split(data, labels)

            branch1 = Tree()
            branch2 = Tree()

            branch1.fit(data, labels)
            branch2.fit(data, labels)

    def find_best_split(self, data, lab):
        for split in data:
            

    def impurity(self, labels):
        p0 = len(labels[np.where(labels == 0)]) / len(labels)
        if p0 == 0 or p0 == 1:
            I = 1
        else:
            I = - p0 * np.log2(p0) - (1 - p0) * np.log2(1 - p0)
        print(I)

    def predict(self, data):
        for row in data:
            pass

    @staticmethod
    def entropy(p_i):
        if p_i == 1 or p_i == 0:
            return 0
        else:
            return - p_i * np.log2(p_i) - (p_i) * np.log2(p_i)

def main():
    seals = Tree(traindata, ground_truth)
    seals.fit(traindata, ground_truth)

if __name__ == "__main__":
    main()
