import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics

# Candidate 25

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
filename_seals_test = os.path.join(dirname, "seals_test.csv")
filename_seals_images_test = os.path.join(dirname, "seals_images_test.csv")
traindata = np.genfromtxt(filename_seals_train, delimiter=" ")
testdata = np.genfromtxt(filename_seals_test, delimiter=" ")
testimages = np.genfromtxt(filename_seals_images_test, delimiter=" ")

class LogisticDiscrimination:
    """
    Class for implementing the logistic discrimination algorithm, tailored for Portfolio assignment 2.
    How to:
        - Initialize object, first calling: model = LogisticDiscrimination(trainingdata)
        - Train model, use: model.train()
        - Test model, use: model.test(testdata)
        - Print performance: print(model)
    """
    def __init__(self, trainingData):
        # Extract ground truth by indices
        self.gt = trainingData[:, 0]

        # Init training data
        self.traindata = np.delete(trainingData, 0, axis=1)
        self.traindata = np.insert(self.traindata, 0, np.ones_like(self.gt), axis=1)

        # Init weights
        self.w = np.random.uniform(-0.01, 0.01, size=(self.traindata.shape[1]))

        # Init derivatives, as array of same shape as weights, but as an random number between -10 and 10
        Dj = np.ones_like(self.w) * np.random.uniform(-10, 10)

        self.has_been_trained = False
        self.test_has_been_called = False

    def sigmoid(self, x, weights):
        """
        The generic sigmoid function. Because of overflow error due to np.exp(-large number) i made a slight modification to use the step before the final, most compact sigmoid, but still the same function.
        """
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
        # Check if this method has been called
        self.has_been_trained = True

        # set stepsize
        s = 0.0003

        # Set gradualDescent to True
        gradualDescent = True
        while gradualDescent:
            # use sigmoid to get y
            y = self.sigmoid(self.traindata, self.w)

            # Caluclate derivatives
            Dj = np.matmul((self.gt - y), self.traindata)

            # Caluclate and update weights
            self.w = self.w + s * Dj

            test_wheter_at_extrema = np.where(Dj < 0.0001, 1, 0)

            # Test if we are at extrema
            if np.all((test_wheter_at_extrema == 1)):
                gradualDescent = False

        # Classify as 0's or 1's by rounding the sigmoid function
        c = np.round(self.sigmoid(self.traindata, self.w))

        # Classify as zeros or ones
        classifiedZero = np.where(c < 0.5)
        classifiedOne = np.where(c > 0.5)

        # Create arrays with ground truth
        actualZero = np.where(self.gt < 0.5)
        actualOne = np.where(self.gt > 0.5)

        # Find and return the performance of the training
        self.trainPerf = self.performance(classifiedZero, classifiedOne, actualZero, actualOne)
        return self.trainPerf

    def test(self, testdata, discriminationThreshold = 0.5):
        """
        Method for testing model on trained weights and biases.
        Args:
            (self) - Instance
            (ndarray) - test data, the data we wish to test. Must have rows the same size as the training data
            (float) - Optional. Thresold for the decision boundary
        Returns:
            (ndarray, ndarray) - (Confusion matrix, accuracy)
        """
        self.test_has_been_called = True

        self.testdata_unchanged = testdata
        # Initialize test data
        groundtruth = testdata[:, 0]
        self.testdata = np.delete(testdata, 0, axis=1)
        self.testdata = np.insert(self.testdata, 0, np.ones_like(groundtruth), axis=1)

        # Classify wheter 0's or 1's depending on the treshold
        classified = np.where(self.sigmoid(self.testdata, self.w) < discriminationThreshold, 0, 1)

        # Same sort of classification as for the training method
        self.classifiedZero = np.where(classified < 0.5)
        self.classifiedOne = np.where(classified > 0.5)

        self.actualZero = np.where(groundtruth < 0.5)
        self.actualOne = np.where(groundtruth > 0.5)

        self.testPerf = self.performance(self.classifiedZero, self.classifiedOne, self.actualZero, self.actualOne)
        return self.testPerf

    def is_model_trained(self):
        """
        Test if model is trained
        """
        try:
            if not self.has_been_trained:
                raise Exception("Cannot test without a trained model!")
        except Exception as e:
            raise e

    def performance(self, classified0, classified1, actual0, actual1):
        """
        method for the performance measures we typically use.
        Args:
            param1: (ndarray) - Array of indexes which are classified as 0's.
            param2: (ndarray) - Array of indexes which are classified as 1's.
            param3: (ndarray) - Array of indexes which are actually 0's.
            param4: (ndarray) - Array of indexes which are actually 1's.
        Returns:
            (ndarray, ndarray) - (confusion matrix, accuracy).
        """
        confmat = self.confusionMatrix(classified0, classified1, actual0, actual1)

        # Calculates the accuracy of the model
        accuracy = (confmat[0][0] + confmat[1][1]) / np.sum(confmat).astype("float32")
        return confmat, accuracy

    def __str__(self):
        """
        Creates a readable print for the viewer of the result of the code with the built in python-magic methods. To use, print the instance of the object, i.e. Foo = LogisticDiscrimination(bar) -> Foo.train() -> Foo.test(testdata) -> print(Foo).
        Returns:
            (str) - To be printed
        """
        # Checks if the model has been trained.
        self.is_model_trained()

        # Creates a string containing the performance of the training.
        string = "Training:\n Confusion matrix:\n {self.trainPerf[0]}\nAccuracy = {self.trainPerf[1]} ".format(self=self)

        if self.test_has_been_called:
            # Run test again to make sure we are using correct decision boundary
            self.test(self.testdata_unchanged)

            # Adds as string containing the performance of the test.
            string += "\n\nTest:\n Confusion matrix:\n {self.testPerf[0]}\nAccuracy = {self.testPerf[1]}".format(self=self)
        return string

    def confusionMatrix(self, classified0, classified1, actual0, actual1):
        # Note: this method has been copied from my own work in Portfolio Assignment 1
        """
        Can be used with the performance method as a measure of how good the algorithm works.
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

    def ROC(self, n=50):
        """
        Function for plotting the ROC, or Reciever Operating Characteristic, curve for the different thresholds for the logistic discrimination model.
        Args:
            param1: (self) - Instance
            param2: (int) - Optional. How many threshold values
        """
        # n Evenely spaced threshold values from 0 to 1
        thresholds = np.linspace(0,  1, n)

        # Array of True positive rates and False positive rates.
        tpRates = np.zeros_like(thresholds)
        fpRates = np.zeros_like(thresholds)

        # Looping trought the threshold array
        for ind, threshold in enumerate(thresholds):

            # Gathering the confusion matrix for every threshold
            confusionmat = self.test(self.testdata_unchanged, threshold)[0]

            tp = confusionmat[0][0]
            fn = confusionmat[0][1]
            fp = confusionmat[1][0]
            tn = confusionmat[1][1]

            # Calculates the rates and fills the arrays
            tpRates[ind] = tp / (tp + fn)
            fpRates[ind] = fp / (fp + tn)

        # my main plot with false postive rates on the first axis, and true positive rates on the second axis.
        mainplot = plt.plot(fpRates, tpRates, label="Logistic discrimination")

        # The diagonal line, generally known as the x/y - line
        xyLine = plt.plot(np.linspace(np.min(fpRates), np.max(fpRates), len(thresholds)), np.linspace(np.min(tpRates), np.max(tpRates), len(thresholds)), "--", label="No discrimination")

        # Using sklearn.metrics AUC method
        AUC = metrics.auc(fpRates, tpRates)

        # Misc for naming the plots.
        title = plt.title("ROC curve")
        xlab = plt.xlabel("False positive rates")
        ylab = plt.ylabel("True positive rates")
        plt.legend(title="n = {} Step size of thresholds\nAUC = {:.3f}".format(n, AUC))
        plt.show()

        return AUC

    def correctly_classified(self):
        """
        Method for extracting 5 correctly classified seal image indices, and 5 wrong classified seal image indices.
        Returns:
            (tuple of arrays) - (ndarray, ndarray), five correct indices, five wrong indices.
        """
        # Run test again to make sure we are using correct
        self.test(self.testdata_unchanged)

        # In essence the same as is done in the confusion matrix, which means find the tp, fn, fp, tn
        correctly_classified_harp = np.intersect1d(self.classifiedZero, self.actualZero)
        correctly_classified_hooded = np.intersect1d(self.classifiedOne, self.actualOne)
        wrongly_classified_harp = np.intersect1d(self.classifiedZero, self.actualOne)
        wrongly_classified_hooded = np.intersect1d(self.classifiedOne, self.actualZero)

        # Add together lists of all correctly classified and all faulty classified into separate arrays
        correct_class = np.concatenate((correctly_classified_harp, correctly_classified_hooded))
        wronged_class = np.concatenate((wrongly_classified_harp, wrongly_classified_hooded))

        # Choose five random of corrects and five random of faulties.
        five_correct = np.random.choice(correct_class, 5)
        five_wrong = np.random.choice(wronged_class, 5)

        return five_correct, five_wrong

def show_image(i, title):
    """
    Show testimage number i
    Args:
        param1: (int) - index of picture
        param2: (str) - title for picture of index i
    """
    # Reshape row i of size 4096 (= 64 x 64)
    plt.imshow(np.reshape(testimages[i], (64, 64)))

    # Add title
    plt.title(str(title))
    plt.show()

def show_five_corr_five_incorr(five_corr, five_wrg):
    """
    Shows 5 correctly classified images & 5 incorretly classified images separately, with sensible title.
    Args:
        param1: (ndarray) - Indices of 5 correctly classified images
        param2: (ndarray) - Indices of 5 incorrectly classified images
    """
    # Combine all images to one array
    all_img = np.concatenate((five_corr, five_wrg))
    for i, index_of_pic in enumerate(all_img):

        # i <= 5 give the correctly classified images
        if i <= 5:

            # Use function show_image to show images
            show_image(index_of_pic, title="Image " + str(i + 1) + ": Correctly classified seal.")
        else:
            show_image(index_of_pic, title="Image " + str(i + 1) + ": Not correctly classified seals.")

def main():
    # Init class
    seals = LogisticDiscrimination(traindata)

    # Train class
    seals.train()

    # Test class
    seals.test(testdata)

    # Print performance of class
    print(seals)

    # Gather samples of correct classifications and misclassified seals
    samples = seals.correctly_classified()

    # Construct ROC curve and AUC
    seals.ROC(100)

    # Show images
    show_five_corr_five_incorr(samples[0], samples[1])

if __name__ == "__main__":
    main()
