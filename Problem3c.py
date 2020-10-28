import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

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
        self.b = None
        self.g = None

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
    def __init__(self, node=None, minimum_impurity=0.6, min_data_nodes=40, max_recursion_depth=5):
        self.minimum_impurity = minimum_impurity
        self.min_data_nodes = min_data_nodes
        self.max_recursion_depth = max_recursion_depth

        # Remember Tree is node through recursion.
        self.Tree = node

    def fit(self, data, gt):
        self.data = data
        self.gt = gt

        self.train(data, gt, self.Tree)

    def train(self, data, gt, node):
        self.data = data
        self.gt = gt

        if self.error(node) < self.tetha:
            node.isLeafnode = True
        else:
            # split data
            # self.bm

            if node.split is None or node.threshold is None:
                node.isLeafnode = True

            # Initialize left and right node
            node.right = Node()
            node.left = Node()

            # Add depth of node
            node.right.depth = node.depth + 1
            node.left.depth = node.depth + 1

            # Calculate entropy of node
            # node.right.nodeclass = self.class_of_node()
            # node.left.nodeclass = self.class_of_node()

            right = Tree(node.right)
            left = Tree(node.left)

            right.train(rightdata, rightgt, node.right)
            left.train(leftdata, leftgt, node.left)


    def error(self, x):
        N = np.sum(self.data * self.bm)
        error = 1 / N * (np.sum(x - self.gm) ** 2 * self.bm) / np.sum(self.bm)

        return error

    def b(self, node):
        for val in self.data:
            g = error(val)
        return b

def main():
    trainingdata = censoredData[:, 0]
    to_be_estimated = censoredData[:, 1]

    reg = RegressionTree()
    reg.fit(trainingdata)

if __name__ == "__main__":
    main()
