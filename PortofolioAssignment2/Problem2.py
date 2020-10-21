import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
filename_seals_test = os.path.join(dirname, "seals_test.csv")
filename_seals_images_test = os.path.join(dirname, "seals_images_test.csv")
traindata = np.genfromtxt(filename_seals_train, delimiter=" ")
testdata = np.genfromtxt(filename_seals_test, delimiter=" ")
# testimages = np.genfromtxt(filename_seals_images_test, delimiter=" ")

class DecisionTree:
    """
    Class for decision tree.
    """
    def __init__(self, trainingData):
        # Extract ground truth by indices
        self.gt = trainingData[:, 0]

        # Init training data
        self.traindata = np.delete(trainingData, 0, axis=1)

        # probability of belonging to class 1, p0 = 1 - p1
        self.p1 = len(self.gt[np.where(self.gt == 1)]) / len(self.gt)

    def train(self):
        best_boundaries = np.zeros_like(self.traindata[0])
        for index, columns in enumerate(self.traindata.T):
            best_boundaries[index] = self.boundary_of_feature(columns)
        print(best_boundaries)

    def boundary_of_feature(self, features):
        # snip away the largest and smallest values as theese would be evaluated to 0 in the for loop
        max_value = np.max(features)
        min_value = np.min(features)

        # features belonging to class C0 and C1
        feature_C0 = features[np.where(self.gt == 0)]
        feature_C1 = features[np.where(self.gt == 1)]

        # Gather entropies in array
        entropies = np.zeros_like(features)
        for i, boundary in enumerate(features):
            if boundary == max_value or boundary == min_value:
                I_tot = 1
            else:
                # Probability of being over the border given beloning to either of the classes.
                p_over_boundary_belongC0 = len(feature_C0[feature_C0 > boundary]) / len(features[features > boundary])

                # Probability of being under the border given beloning to eiter of the classes.
                p_under_boundary_belongC0 = len(feature_C0[feature_C0 <= boundary]) / len(features[features <= boundary])

                # Total over the boundary
                total_over_boundary = len(features[features > boundary]) / len(features)

                # One minus this gives under boundary
                total_under_boundary = 1 - total_over_boundary

                I_tot = total_over_boundary * self.entropy(p_over_boundary_belongC0) + total_under_boundary * self.entropy(p_under_boundary_belongC0)

            entropies[i] = I_tot

        best_boundary_value = features[np.where(entropies == np.min(entropies))][0]
        best_boundary_entropy = np.min(entropies)
        best_boundary = np.where(entropies == np.min(entropies))[0][0]
        # print("boundary value = {}, entropy = {}, index = {}".format(best_boundary_value, best_boundary_entropy, best_boundary))
        return best_boundary_value

    def entropy(self, p_i):
        if p_i == 1 or p_i == 0:
            return 0
        else:
            return - p_i * np.log2(p_i) - (p_i) * np.log2(p_i)

def main():
    seals = DecisionTree(traindata)
    seals.train()

if __name__ == "__main__":
    main()
