import numpy as np
import os

dirname = os.path.dirname(__file__)
filename_tictac = os.path.join(dirname, "tictac-end.csv")

# Import and initialize the data
data = np.genfromtxt(filename_tictac, delimiter=" ", dtype="int64")
np.random.shuffle(data)

# How many percent of our data we would like to use for training the algorithm
trainingpercent = 0.7
trainingpercentdata = np.floor(data.shape[0] * trainingpercent).astype("int64")

# Training data is the first 70% of the whole dataset, then test is the remainder
trainingData = data[0 : trainingpercentdata]
testdata = data[trainingpercentdata - 1 : -1]

# Storing the ground truth in a separate array
groundTruth = trainingData[:, 0]

# Choosing the relevant data by adding a column of ones where we remove the ground truth column.
onearr = np.ones_like(groundTruth)
tactoes = np.delete(trainingData, 0, axis=1)
tactoes = np.insert(tactoes, 0, onearr, axis=1)

# Define a sigmoid function.
sigmoid = lambda x, weights: 1 / (1 + np.exp(-(np.matmul(x, weights))))

def gradDescent(X, r):
    # Define random vector of weights with the same dimension as our data, 9 in this case
    w = np.random.uniform(-0.01, 0.01, size=(X.shape[1])) #.astype("float128")

    # Step size, may change later.
    s = 0.08

    # Create array of ones with same shape as the weights this is the derivatives
    Dj = np.ones_like(w) * np.random.uniform(-10, 10)

    grddesc = True
    while grddesc:
        y = sigmoid(X, w)
        Dj = np.matmul((r - y), X)
        w = w + s * Dj

        test = np.where(Dj < 0.01, 1, 0)

        # Test when convergence
        if np.all((test == 1)):
            grddesc = False

    c = np.round(sigmoid(X, w))
    return c

classification = gradDescent(tactoes, groundTruth)

classifiedZero = np.where(classification < 0.5)
classifiedOne = np.where(classification > 0.5)

actualZero = np.where(groundTruth < 0.5)
actualOne = np.where(groundTruth > 0.5)

def confusionMatrix(classified0, classified1, actual0, actual1):
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

confusionmatrix = confusionMatrix(classifiedZero, classifiedOne, actualZero, actualOne)

accuracy = (confusionmatrix[0][0] + confusionmatrix[1][1]) / np.sum(confusionmatrix).astype("float32")

print("This algorithm gives the confusion matrix: \n{:},\n and has an accuracy of {:}.".format(confusionmatrix, accuracy))
