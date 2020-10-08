import numpy as np
import os
dirname = os.path.dirname(__file__)
filename_tictac = os.path.join(dirname, "tictac-end.csv")

# Import and initialize the data
data = np.genfromtxt(filename_tictac, delimiter=" ", dtype="int64")
np.random.shuffle(data)

groundTruth = data[:, 0]
tactoes = np.delete(data, 0, axis=1)

sigmoid = lambda x, weights, bias: 1 / (1 + np.exp(-(np.matmul(x, weights) + bias)))

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

def gradDescent(X, r):
    # Define random vector of weights with the same dimension as our data, 9 in this case
    # w = np.ones(1)
    # w = np.concatenate((w, np.random.uniform(-0.01, 0.01, size=(X.shape[1]))))

    w = np.random.uniform(-0.01, 0.01, size=(X.shape[1]))
    # Define random bias
    w0 = np.random.uniform(-0.01, 0.01)

    # Step size, may change later.
    s = 5

    # Iterate throught each result (row of data)
    for t in range(0, np.shape(X)[0]):
        # Create array of ones with same shape as the weights this is the derivatives
        Dj = np.ones_like(w) * 10
        grddesc = True

        while grddesc:

            y = sigmoid(X, w, w0)
            Dj = (r[t] - y[t]) * X[t]
            w = w + s * Dj
            w0 = w0 + s * r[t] - y[t]
            test = np.where(Dj < 0.01, 1, 0)

            # Test when convergence
            if np.all((test == 1)):
                grddesc = False

    c = np.round(sigmoid(X, w, w0))

    classifiedZero = np.where(c < 0.5)
    classifiedOne = np.where(c > 0.5)

    actualzero = np.where(r < 0.5)
    actualone = np.where(r > 0.5)

    confusionmatrix = confusionMatrix(classifiedZero, classifiedOne, actualzero, actualone)

    print(np.sum(confusionmatrix), confusionmatrix[0][0], confusionmatrix[1][1])

    accuracy = (confusionmatrix[0][0] + confusionmatrix[1][1]) / np.sum(confusionmatrix)

    print(accuracy)
gradDescent(tactoes, groundTruth)
