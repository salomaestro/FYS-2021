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

        # Assigned when find_best_split function is called. This is the index and threshold which made this a node.
        self.split = None
        self.threshold = None

        self.gm = None
        self.bm = None

        # Either node can be 0, or node can be 1, stored here.
        self.nodeclass = None

        self.errorval = None
        # Is declared leaf if True
        self.isLeafnode = False

        # Node's depth
        self.depth = None

        # if node is a leaf, stores the ratios of ones to zeros, such that the classifier can change it's decision boundary.
        self.ratio = None

class RegressionTree:
    """
    Regression tree!
    """
    def __init__(self, node=None, tetha = 1, min_data_nodes=10, max_recursion_depth=10):
        self.tetha = tetha
        self.min_data_nodes = min_data_nodes
        self.max_recursion_depth = max_recursion_depth

        # Remember Tree is node through recursion.
        self.Tree = node

    def fit(self, data, gt):
        bm = np.ones_like(gt)
        gm = np.mean(gt)

        self.Tree = Node()
        self.Tree.depth = 1

        self.train(data, gt, gm, bm, self.Tree)

    def train(self, data, gt, gm, bm, node):
        node.gm = gm
        node.bm = bm
        #print("depth =", node.depth)

        if self.isLeaf(data, gt, node, bm, gm):
            node.isLeafnode = True
            #print("i found a leaf", node.bm, "mean =", node.gm)
            return
        else:
            # split data
            # self.bm
            threshold, split_index, lowest_error = self.findsplit(data, gt, gm, node)

            node.errorval = lowest_error

            node.split = split_index
            node.threshold = threshold

            if node.split is None or node.threshold is None:
                node.isLeafnode = True
                #print("i found a leaf", node.bm, "mean =", node.gm)
                return

            rightbm, leftbm = self.split(data, gt, threshold, node)

            # Initialize left and right node
            node.right = Node()
            node.left = Node()

            # Add depth of node
            node.right.depth = node.depth + 1
            node.left.depth = node.depth + 1

            # remember the b array for node
            node.right.bm = rightbm
            node.left.bm = leftbm

            node.right.gm = self.calc_mean(data, gt, rightbm)
            node.left.gm = self.calc_mean(data, gt, leftbm)

            #print("create branch")

            right = RegressionTree(node.right)
            left = RegressionTree(node.left)

            right.train(data, gt, node.right.gm, rightbm, node.right)
            left.train(data, gt, node.left.gm, leftbm, node.left)

    def isLeaf(self, data, gt, node, bm, gm):
        """
        Method to check if node is a leaf or not, with branches data.

        Args:
            param1: (np.ndarray), trainingdata of shape (n, m) with features n (columns), and m datapoints (rows).
            param2: (np.ndarray), ground truth, should be of size gt.size = (n,).
            param3: (Node-object), current node.
        Returns:
            (bool), Either True or False depending on if node is leaf or not.
        """
        # Declare leaf
        if node.depth >= self.max_recursion_depth:
            return True
        elif self.error(bm, data, gt, gm) < self.tetha:
            return True
        elif len(bm) < self.min_data_nodes:
            return True
        else:
            return False

    def calc_mean(self, data, gt, bm):
        g = np.sum(bm * data * gt) / np.sum(bm * data)
        return g

    def error(self, b, data, gt, gm):
        N = np.sum(b)
        error = 1 / N * (np.sum((gt - gm) ** 2 * b)) / np.sum(b)
        return error

    def findsplit(self, data, gt, gm, node):
        splits = []
        thresholds = []
        data = data * node.bm
        max = np.max(data)
        min = np.min(data)
        for val in data:
            if val == max or val == min:
                thresholds.append(1)
            else:
                possible_b = np.where(data <= val, 1, 0)
                splits.append(self.error(possible_b, data, gt, gm))
                thresholds.append(val)

        if len(splits) < 2:
            return None, None, None

        lowest_error = np.min(splits)
        split_index = np.where(splits == lowest_error)[0][0]
        threshold = thresholds[split_index]
        #print("threshold =",threshold, "split index =", split_index)

        return threshold, split_index, lowest_error

    def split(self, data, gt, threshold, node):
        rightsplit, leftsplit = [], []
        data = data * node.bm
        # data = data[np.where(data == 1)]

        for i, val in enumerate(data):
            if val <= threshold:
                rightsplit.append(1)
                leftsplit.append(0)
            else:
                leftsplit.append(1)
                rightsplit.append(0)

        rightsplit = np.asarray(rightsplit)
        leftsplit = np.asarray(leftsplit)

        #print("size of split. right:", len(np.where(rightsplit == 1)[0]), "left:", len(np.where(leftsplit == 1)[0]))

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
            param1: (np.ndarray), row from data
            param2: (Node-object), particular node, first time called it is the main tree.
        returns:
            (int), Either 0 or 1, depending on classification.
        """
        # Check if particular node is a leaf node, or if the threshold does not exist.
        if node.isLeafnode or node.threshold is None:
            return node.gm

        # Recursively move down the tree until each leaf nodes prediction has been returned.
        if value > node.threshold:
            prob = self.predict_sample(value, node.right)
        else:
            prob = self.predict_sample(value, node.left)
        return prob

def main():
    trainingdata = censoredData[:, 0]
    gt = censoredData[:, 1]
    gt = np.delete(gt, index_of_nan)
    trainingdata = np.delete(trainingdata, index_of_nan)
    testdata = gt[index_of_nan]

    reg = RegressionTree()
    reg.fit(trainingdata, gt)
    predicted_data = reg.predict(testdata)
    comparedata = uncensoredData[:, 1]
    comparedata = comparedata[index_of_nan]
    MSE_regressionTree_estimator = np.mean((comparedata - predicted_data) ** 2)
    print("Mean Sqaured Error for Regression Tree estimator =", MSE_regressionTree_estimator)

if __name__ == "__main__":
    main()
