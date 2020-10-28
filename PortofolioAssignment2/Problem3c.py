import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from Problem3_ab import *

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

        self.bm = None
        self.bm = None

        # Either node can be 0, or node can be 1, stored here.
        self.nodeclass = None

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
    def __init__(self, node=None, tetha = 0.01, min_data_nodes=40, max_recursion_depth=5):
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

        if self.isLeaf(data, gt, node, bm):
            node.isLeafnode = True
            print("i found a leaf")
            return
        else:
            # split data
            # self.bm
            threshold, split_index, lowest_error = self.findsplit(data, gt, gm)
            rightdata, leftdata, rightgt, leftgt = self.split(data, gt, threshold)

            node.split = split_index
            node.threshold = threshold

            if node.split is None or node.threshold is None:
                node.isLeafnode = True
                print("i found a leaf")
                return

            # Initialize left and right node
            node.right = Node()
            node.left = Node()

            bright = np.zeros_like(bm)
            bleft = np.zeros_like(bm)
            print(bleft.shape, bright.shape)

            bright[rightdata] = 1
            bleft[leftdata] = 1

            rightdata = data[rightdata]
            leftdata = data[leftdata]

            # Add depth of node
            node.right.depth = node.depth + 1
            node.left.depth = node.depth + 1

            # Calculate entropy of node
            node.right.bm = bm
            node.left.bm = bm
            print("create branch")

            right = RegressionTree(node.right)
            left = RegressionTree(node.left)

            right.train(rightdata, rightgt, gm, bright, node.right)
            left.train(leftdata, leftgt, gm, bleft, node.left)

    def isLeaf(self, data, gt, node, bm):
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
        elif self.error(bm, data, gt, bm) < self.tetha:
            return True
        elif len(data) < self.min_data_nodes:
            return True
        else:
            return False


    def error(self, b, data, gt, gm):
        N = np.sum(b)
        error = 1 / N * (np.sum((gt - gm) ** 2 * b)) / np.sum(b)
        return error

    def findsplit(self, data, gt, gm):
        splits = []
        thresholds = []
        max = np.max(data)
        min = np.min(data)
        for val in data:
            if val == max or val == min:
                thresholds.append(1)
            else:
                possible_b = np.where(data <= val, 1, 0)
                splits.append(self.error(possible_b, data, gt, gm))
                thresholds.append(val)

        lowest_error = np.min(splits)
        split_index = np.where(splits == lowest_error)[0][0]
        threshold = thresholds[split_index]
        print(threshold, split_index)

        return threshold, split_index, lowest_error

    def split(self, data, gt, threshold):
        split1, split2 = [], []
        split1gt, split2gt = [], []

        for i, val in enumerate(data):
            if val <= threshold:
                split1.append(i)
                split1gt.append(gt[i])
                split2gt.append(0)
            else:
                split2.append(i)
                split2gt.append(gt[i])
                split1gt.append(0)

        print("here", len(split1), len(split2))

        return np.asarray(split1), np.asarray(split2), np.asarray(split1gt), np.asarray(split2gt)

def main():
    trainingdata = censoredData[:, 0]
    gt = censoredData[:, 1]
    gt = np.delete(gt, index_of_nan)
    trainingdata = np.delete(trainingdata, index_of_nan)

    reg = RegressionTree()
    reg.fit(trainingdata, gt)

if __name__ == "__main__":
    main()
