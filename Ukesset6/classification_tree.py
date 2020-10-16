import numpy as np

class Tree:
    """
    Method for classifying with a decision tree approach
    """
    def __init__(self, dataset):
        # Data structured as ndarray([groundtruth, feature1, feature2, ... featureN])
        self.data = dataset
        np.random.shuffle(self.data)
        trainingpercent = 0.7
        trainingpercentdata = np.floor(self.data.shape[0] * trainingpercent).astype("int64")
        self.trainingData = self.data[0 : trainingpercentdata]
        self.tesData = self.data[trainingpercentdata - 1 : -1]
        self.ground_truth = self.trainingData[:, 0]

        # Settings
        self.minimum_impurity = 1

        # Init node
        self.firstNode = np.random.uniform()


    def impurity(self, probMat):
        I = - np.sum(probMat * np.log2(probMat))
        return I

    def total_entropy(self, probMat):
        # Total entropy/impurity after the split
        pass

    def fit(self):
        if impurity(self.data) < minimum_impurity
            # declare leaf node
