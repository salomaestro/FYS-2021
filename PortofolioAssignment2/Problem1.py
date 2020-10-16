import numpy as np
import matplotlib.pyplot as plt
import os

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
data = np.genfromtxt(filename_seals_train, delimiter=" ")

class LogisticDiscrimination:
    """
    Class for implementing the logistic discrimination algorithm, tailored for Portfolio assignment 2.
    """
    def __init__(self, trainingData):
        # Quick init, will have to initialize trainig and test data later on in the methods.
        self.data = trainingData

    # Chose to use lambda function since its so small
    # sigmoid = lambda self, x, weights: 1 / (1 + np.exp(-(np.matmul(x, weights))))
    def sigmoid(self, x, weights):
        mul = np.matmul(x, weights)
        return np.exp(mul) / (1 + np.exp(mul))

    def train(self):
        """
        Method for training model with training data.
        """
        # Extract ground truth by indices
        self.gt = self.data[:, 0]

        # Init training data
        self.traindata = np.delete(self.data, 0, axis=1)
        self.traindata = np.insert(self.traindata, 0, np.ones_like(self.gt), axis=1)

        # Init weights
        self.w = np.random.uniform(-0.01, 0.01, size=(self.traindata.shape[1]))

        # Init derivatives, as array of same shape as weights, but as an random number between -10 and 10
        Dj = np.ones_like(self.w) * np.random.uniform(-10, 10)

        # set stepsize
        s = 0.0005

        # Set gradualDescent to True
        gradualDescent = True
        while gradualDescent:
            # use sigmoid to get y
            print(np.matmul(self.traindata, self.w))
            y = self.sigmoid(self.traindata, self.w)
            Dj = np.matmul((self.gt - y), self.traindata)
            print(Dj)
            self.w += s * Dj

            test_wheter_at_extrema = np.where(Dj < 0.5, 1, 0)

            # Test if we are at extrema
            if np.all((test_wheter_at_extrema == 1)):
                gradualDescent = False

        # Classify as 0's or 1's
        self.c = np.round(self.sigmoid(self.traindata, self.w))

        classifiedZero = np.where(self.c < 0.5)
        classifiedOne = np.where(self.c > 0.5)

        actualZero = np.where(self.gt < 0.5)
        actualOne = np.where(self.gt > 0.5)

        return classifiedZero, classifiedOne

    def test(self, testdata):
        """
        Method for testing model on trained weights and biases.
        """
        self.testdata = testdata

    def confusionMatrix(self, classified0, classified1, actual0, actual1):
        # Note: this method has been copied from my own work in Portfolio Assignment 1
        """
        A measure of how good the algorithm works.
        Args:
            param1: (ndarray) - Array of indexes which are classified as 0's.
            param2: (ndarray) - Array of indexes which are classified as 1's.
            param3: (ndarray) - Array of indexes which are actually 0's.
            param4: (ndarray) - Array of indexes which are actually 1's.
        Returns:
            (ndarray) - Nested list which is the confusion matrix.
        """
        # intersect1d compares two arrays and returns the elements which are found in both arrays.
        tp = np.intersect1d(classified0, actual0).shape[0]
        fn = np.intersect1d(classified1, actual0).shape[0]
        fp = np.intersect1d(classified0, actual1).shape[0]
        tn = np.intersect1d(classified1, actual1).shape[0]
        return np.array([np.array([tp, fn]), np.array([fp, tn])])


def main():
    seals = LogisticDiscrimination(data)
    print(seals.train())

if __name__ == "__main__":
    main()
