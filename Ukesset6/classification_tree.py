import numpy as np
import os

dirname = os.path.dirname(__file__)
filename_blobs = os.path.join(dirname, "blobs.csv")

# Import and initialize the data
data = np.genfromtxt(filename_blobs, delimiter=" ", dtype="float32")

def exctractFromData(dataset, trainpercent=0.7):
    # How many percent of our data we would like to use for training the algorithm
    trainpercentdata = np.floor(dataset.shape[0] * trainpercent).astype("int64")

    # Training data is the first x % of the whole dataset, then test is the remainder
    trainData = dataset[0 : trainpercentdata]
    testData = dataset[trainpercentdata - 1 : -1]

    groundTruth = trainData[:, 0].astype("int64")
    # Remove ground truth from trainingdata
    trainData = np.delete(trainData, 0, axis=1)
    testData = np.delete(testData, 0, axis=1)

    return trainData, testData, groundTruth

trainingData, testingData, groundTruth = exctractFromData(data)



c0 = data[np.where(data[:, 0] == 0)]
c1 = data[np.where(data[:, 0] == 1)]

splits_attr1 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), num=100)
splits_attr2 = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), num=100)



class Tree:
    """
    Method for classifying with a decision tree approach
    """
    def __init__(self, train, test, ground_truth, data):
        # init data, training, test and ground truth.
        self.train = train
        self.test = test
        self.gt = ground_truth
        self.data = data

        # Settings
        self.minimum_impurity = 1

        # # Init node
        # self.firstNode = np.random.uniform()

    def splitAttribute(self):
        c0 = data[np.where(data[:, 0] == 0)]
        c1 = data[np.where(data[:, 0] == 1)]

        splits_attr1 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), num=100)
        splits_attr2 = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), num=100)

        entropy = split_entropy(splits_attr1, splits_attr2)



    def split_entropy(self, probMat):
        # Total entropy/impurity after the split
        impurity = np.sum(self.data)


    def nodeEntropy(self, probMat):
        I = - np.sum(probMat * np.log2(probMat))
        return I
