import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from Problem3_ab import *
import sys

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_censored_data = os.path.join(dirname, "censored_data.csv")
filename_uncensored_data = os.path.join(dirname, "uncensored_data.csv")

censoredData = np.genfromtxt(filename_censored_data, delimiter=" ")
uncensoredData = np.genfromtxt(filename_uncensored_data, delimiter=" ")
censoredData_df = pd.DataFrame(censoredData)

class Node:
    """
    Object for each node of Decision Tree object.
    """
    def __init__(self):
        # left and right nodes to keep track of the direction of each node.
        self.right = None
        self.left = None

        # Assigned when findsplit function is called. This is the threshold which made this a node.
        self.threshold = None

        # Mean at node
        self.gm = None

        # Array of b at node
        self.bm = None

        # Is declared leaf if True
        self.isLeafnode = False

        # Node's depth
        self.depth = None


class RegressionTree:
    """
    Regression tree!
    Play around with tetha, min data size and the max recursion depth to see how the tree adapts.

    Args:
        param1: (float) - optional, standard as 0.06, a float value such that 0 < tetha < 1
        param2: (int) - optional, standard as 5, integer specifying how deep the tree is allowed to go.
        param3: (Node-object) - Should not be set as anything!
    """
    def __init__(self, tetha = 0.06, max_recursion_depth=3, node=None):
        self.tetha = tetha
        self.max_recursion_depth = max_recursion_depth

        # Remember Tree is node through recursion.
        self.Tree = node

    def fit(self, data, rt):
        """
        Generic fit function for object. This is the method which should be called to fit and train the model.
        -Calls train method to begin trainig.

        Args:
            param1: (np.ndarray) - first column which is should be correlated to param2.
            param2: (np.ndarray) - rt, secound column, which will be learned.
        """
        # At first b array will be an array with shape equal to rt, filled with only ones
        bm = np.ones_like(rt)

        # First run of tree mean is for the whole tree
        gm = np.mean(rt)

        # Init tree and set depth as one
        self.Tree = Node()
        self.Tree.depth = 1

        # Begin training and creating Tree object.
        self.train(data, rt, gm, bm, self.Tree)

    def train(self, data, rt, gm, bm, node):
        """
        Part of object's fit method. should not be used to train the model, instead call object.fit(Args)

        Args:
            param1: (np.ndarray) - first column which is should be correlated to param2.
            param2: (np.ndarray) - rt, secound column, which will be learned.
            param3: (float) - mean of set.
            param4: (np.ndarray) - Array of ones and zeros, one indicates that datapoint x has reached node m, 0 otherwize
            param5: (Node-object) - Keeps track of branch and builds Tree object.
        """
        # Check if node is leaf
        if self.isLeaf(data, rt, node, bm, gm):
            node.isLeafnode = True
            return
        else:
            # Find the best split in the dataset
            threshold = self.findsplit(data, rt, gm, node)
            print(threshold)
            # Assign nodes split index and threshold
            node.threshold = threshold

            # Declare leaf node if any of theese are still None
            if node.threshold is None:
                node.isLeafnode = True
                return

            # Split the b array into right and left b arrays using the threshold found in findsplit function
            rightbm, leftbm = self.split(data, threshold)

            # Initialize left and right node
            node.right = Node()
            node.left = Node()

            # Add depth of node
            node.right.depth = node.depth + 1
            node.left.depth = node.depth + 1

            # remember the b array for node right and left
            node.right.bm = rightbm
            node.left.bm = leftbm

            # Calculate the mean at
            node.right.gm = self.calc_mean(rt, rightbm)
            node.left.gm = self.calc_mean(rt, leftbm)

            # Initialize right and left branch of tree, and pass in the respective Node objects.
            right = RegressionTree(node = node.right)
            left = RegressionTree(node = node.left)

            # Recursively train model, with left and right branches.
            right.train(data, rt, node.right.gm, node.right.bm, node.right)
            left.train(data, rt, node.left.gm, node.left.bm, node.left)

    def isLeaf(self, data, rt, node, bm, gm):
        """
        Method to check if node is a leaf or not, with branches data.

        Args:
            param1: (np.ndarray) - first column which is should be correlated to param2.
            param2: (np.ndarray) - rt, secound column, which will be learned.
            param3: (Node-object) - current node.
            param4: (np.ndarray) - Array of ones and zeros, one indicates that datapoint x has reached node m, 0 otherwize
            param5: (float) - mean of set.
        Returns:
            (bool), Either True or False depending on if node is leaf or not.
        """
        # Declare leafs
        if node.depth >= self.max_recursion_depth:
            return True
        elif self.error(bm, data, rt, gm) < self.tetha:
            return True
        else:
            return False

    def calc_mean(self, rt, bm):
        """
        Method to calculate mean of node.

        Args:
            param1: (np.ndarray) - rt, secound column, which will be learned.
            param2: (np.ndarray) - Array of ones and zeros, one indicates that datapoint x has reached node m, 0 otherwise
        Returns:
            (float) - mean of set.
        """
        g = np.sum(bm * rt) / np.sum(bm)
        return g

    def error(self, b, data, rt, gm):
        """
        Method to calculate the error.

        Args:
            param1: (np.ndarray) - Array of ones and zeros, one indicates that datapoint x has reached node m, 0 otherwize
            param2: (np.ndarray) - first column which is should be correlated to param2.
            param3: (np.ndarray) - rt, secound column, which will be learned.
            param4: (float) - mean of set.
        Returns:
            (float) - Error value
        """
        N = np.sum(b)

        # Ensure we dont have any division by 0
        if N == 0:
            return 1000
        else:
            return 1 / N * (np.sum((rt - gm) ** 2 * b))

    def findsplit(self, data, rt, gm, node):
        """
        Method to find best split index and it's threshold value.

        Args:
            param1: (np.ndarray) - first column which is should be correlated to param2.
            param2: (np.ndarray) - rt, secound column, which will be learned.
            param3: (float) - mean of set.
            param4: (Node-object) - current node.
        Returns:
            (float, int) - (threshold, split index)
        """
        splits = []
        thresholds = []

        # Ensure node.bm even have been assigned b array
        if node.bm is not None:

            # Reduce dataset to only those values we are interested in
            data = data * node.bm

        max = np.max(data)
        min = np.min(data)
        for val in data:
            # Ensure the max and min values wont get the best error, since this would yield an empty b array
            if val == max or val == min:
                splits.append(1000)
                thresholds.append(val)
            else:
                # Create array of possible splits.
                possible_b = np.where(data <= val, 1, 0)

                # Calculate all errors of possible splits.
                splits.append(self.error(possible_b, data, rt, gm))
                thresholds.append(val)

        # Filter out the split with the lowest error
        lowest_error = np.min(splits)

        # Find at which index this is
        split_index = np.where(splits == lowest_error)[0][0]

        # Find at which threshold that split index is
        threshold = thresholds[split_index]

        return threshold

    def split(self, data, threshold):
        """
        Method to split b array, such that one is the inverse of the other, i.e where there are ones in the right split, there are zeros in the left split.

        Args:
            param1: (np.ndarray) - first column which is should be correlated to param2.
            param2: (float) - threshold calculated in findsplit method.
        Returns:
            (np.ndarray, np.ndarray) - (b array right, b array left), splitted b arrays
        """
        # Assigns values from data, larger or smaller threshold as 1 or 0.
        rightsplit = np.where(data <= threshold, 1, 0)
        leftsplit = np.where(data > threshold, 1, 0)

        return rightsplit, leftsplit

    def predict(self, testdata):
        """
        Method for classifying data after Tree has been trained.

        Args:
            param1: (np.ndarray), testing data
        Returns:
            (np.ndarray), predicted 0's and 1's.
        """
        predictions = []
        for value in testdata:
            # This function takes the Tree object itself in, since this has stored all nodes, leafs, etc.
            prob = self.predict_sample(value, self.Tree)
            predictions.append(prob)
        return np.asarray(predictions)

    def predict_sample(self, value, node):
        """
        Method for predicting a particular sample. This method should not be called alone, but through the predict method.

        Args:
            param1: (np.ndarray), value from data
            param2: (Node-object), particular node, first time called it is the main tree.
        returns:
            (int), Either 0 or 1, depending on classification.
        """
        # Check if particular node is a leaf node, or if the threshold does not exist.
        if node.isLeafnode or node.threshold is None:

            # return mean of sample
            return node.gm

        # Recursively move down the tree until each leaf nodes prediction has been returned.
        if value <= node.threshold:
            prob = self.predict_sample(value, node.right)
        else:
            prob = self.predict_sample(value, node.left)
        return prob

def main():
    # trainingdata is a combination of first column, which we know has a strong correlation with the middle column, and the middle column, which is stored in variable rt
    trainingdata = censoredData[:, 0]
    rt = censoredData[:, 1]

    # Remove datapoints which is at index which middle coulmn hosts a nan value
    rt = np.delete(rt, index_of_nan)
    trainingdata = np.delete(trainingdata, index_of_nan)

    # testingdata will be the first column of censoredData at indexes which holds nan values
    testdata = censoredData[:, 0][index_of_nan]

    # Initialize regression tree object
    reg = RegressionTree()

    # Train model
    reg.fit(trainingdata, rt)

    # Predict what nan values should be
    predicted_data = reg.predict(testdata)

    # fix data to compare against, the calculate the Means Squared Error of the regression tree
    comparedata = uncensoredData[:, 1]
    comparedata = comparedata[index_of_nan]
    MSE_regressionTree_estimator = np.mean((comparedata - predicted_data) ** 2)
    print("Mean Sqaured Error for Regression Tree estimator =", MSE_regressionTree_estimator)

if __name__ == "__main__":
    main()
