import numpy as np
import matplotlib.pyplot as plt
import os

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
filename_seals_test = os.path.join(dirname, "seals_test.csv")
traindata = np.genfromtxt(filename_seals_train, delimiter=" ")
testdata = np.genfromtxt(filename_seals_test, delimiter=" ")

class LogisticDiscrimination:
    """
    Class for implementing the logistic discrimination algorithm, tailored for Portfolio assignment 2.
    """
    def __init__(self, trainingData, stepsize=0.0003):
        # Extract ground truth by indices
        self.gt = trainingData[:, 0]

        # Init training data
        self.traindata = np.delete(trainingData, 0, axis=1)
        self.traindata = np.insert(self.traindata, 0, np.ones_like(self.gt), axis=1)

        # Init weights
        self.w = np.random.uniform(-0.01, 0.01, size=(self.traindata.shape[1]))

        # Init derivatives, as array of same shape as weights, but as an random number between -10 and 10
        Dj = np.ones_like(self.w) * np.random.uniform(-10, 10)

        self.test_has_been_called = False

    # Chose to use lambda function since its so small
    # sigmoid = lambda self, x, weights: 1 / (1 + np.exp(-(np.matmul(x, weights))))
    def sigmoid(self, x, weights):
        mul = np.matmul(x, weights)
        return np.exp(mul) / (1 + np.exp(mul))

    def train(self):
        """
        Method for training model with training data.
        Args:
            (self) - Instance, training data has been initialized in the __init__ function.
        Returns:
            (ndarray, ndarray) - (Confusion matrix, accuracy)
        """
        # set stepsize
        s = 0.0003

        # Set gradualDescent to True
        gradualDescent = True
        runs = 0
        while gradualDescent:
            # use sigmoid to get y
            y = self.sigmoid(self.traindata, self.w)
            Dj = np.matmul((self.gt - y), self.traindata)
            self.w = self.w + s * Dj

            test_wheter_at_extrema = np.where(Dj < 0.05, 1, 0)

            # Test if we are at extrema
            if np.all((test_wheter_at_extrema == 1)):
                gradualDescent = False
            runs += 1
        # Classify as 0's or 1's
        c = np.round(self.sigmoid(self.traindata, self.w))

        classifiedZero = np.where(c < 0.5)
        classifiedOne = np.where(c > 0.5)

        actualZero = np.where(self.gt < 0.5)
        actualOne = np.where(self.gt > 0.5)
        self.trainPerf = self.performance(classifiedZero, classifiedOne, actualZero, actualOne)
        return self.trainPerf

    def test(self, testdata):
        """
        Method for testing model on trained weights and biases.
        Args:
            (self) - Instance
            (ndarray) - test data, the data we wish to test. Must have rows the same size as the training data
        Returns:
            (ndarray, ndarray) - (Confusion matrix, accuracy)
        """
        self.test_has_been_called = True
        groundtruth = testdata[:, 0]
        self.testdata = np.delete(testdata, 0, axis=1)
        self.testdata = np.insert(self.testdata, 0, np.ones_like(groundtruth), axis=1)

        classified = np.round(self.sigmoid(self.testdata, self.w))
        classifiedZero = np.where(classified < 0.5)
        classifiedOne = np.where(classified > 0.5)

        actualZero = np.where(groundtruth < 0.5)
        actualOne = np.where(groundtruth > 0.5)
        self.testPerf = self.performance(classifiedZero, classifiedOne, actualZero, actualOne)
        return self.testPerf

    def performance(self, classified0, classified1, actual0, actual1):
        confmat = self.confusionMatrix(classified0, classified1, actual0, actual1)
        accuracy = (confmat[0][0] + confmat[1][1]) / np.sum(confmat).astype("float32")
        return confmat, accuracy

    def __str__(self):
        string = "Training:\n Confusion matrix:\n {self.trainPerf[0]}\nAccuracy = {self.trainPerf[1]} ".format(self=self)
        if self.test_has_been_called:
            string += "\n\nTest:\n Confusion matrix:\n {self.testPerf[0]}\nAccuracy = {self.testPerf[1]}".format(self=self)
        return string

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
    seals = LogisticDiscrimination(traindata)
    seals.train()
    seals.test(testdata)
    print(seals)

if __name__ == "__main__":
    main()
