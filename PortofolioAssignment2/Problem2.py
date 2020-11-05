import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn import metrics

# Candidate 25

if sys.version_info[0] != 3:
    print("This code might not be working properly because you are running python version {}, while a python 3 distribution is preferred!".format(sys.version))

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
filename_seals_test = os.path.join(dirname, "seals_test.csv")

traindata = np.genfromtxt(filename_seals_train, delimiter=" ")
testdata = np.genfromtxt(filename_seals_test, delimiter=" ")

# Filter out the ground truth from the training and test set.
ground_truth = traindata[:, 0]
traindata = np.delete(traindata, 0, axis=1)

test_gt = testdata[:, 0]
testdata = np.delete(testdata, 0, axis=1)

class Node:
    """
    Object for each node of Decision Tree object.
    """
    def __init__(self):
        # left and right nodes to keep track of the direction of each node.
        self.right = None
        self.left = None

        # Assigned when find_best_split function is called. This is the index and threshold which made this a node.
        self.split_index = None
        self.threshold = None

        # Either node can be 0, or node can be 1, stored here.
        self.nodeclass = None

        # Is declared leaf if True
        self.isLeafnode = False

        # Node's depth
        self.depth = None

        # if node is a leaf, stores the ratios of ones to zeros, such that the classifier can change it's decision boundary.
        self.ratio = None

class Tree:
    """
    Class for decision tree.

    Args:
        param1: (Node-object), optional, standard set as None
        param2: (float), optional, standard set as 0.6
        param3: (int), optional, standard set as 40
        param4: (int), optinal, standard set as 5
    """
    def __init__(self, node=None, minimum_impurity=0.6, min_data_nodes=40, max_recursion_depth=10):
        # minimum_impurity=0.6, min_data_nodes=40, max_recursion_depth=5
        self.minimum_impurity = minimum_impurity
        self.min_data_nodes = min_data_nodes
        self.max_recursion_depth = max_recursion_depth

        # Remember Tree is node through recursion.
        self.Tree = node

    def fit(self, data, gt):
        """
        Generic fit function for object

        Args:
            param1: (np.ndarray), trainingdata of shape (n, m) with features n (columns), and m datapoints (rows).
            param2: (np.ndarray), ground truth, should be of size gt.size = (n,).
        Returns:
            (list), inherits return from train method.
        """
        # Initialize tree as Node object.
        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.nodeclass = self.class_of_node(gt)

        # Call the train method of Tree object.
        return self.train(data, gt, self.Tree)

    def train(self, data, gt, node, ratios=list()):
        """
        Part of object's fit method. should not be used to train the model, instead call object.fit(Args)

        Args:
            param1: (np.ndarray), trainingdata, as in object's fit method.
            param2: (np.ndarray), ground truth as in object's fit method.
            param3: (Node-object), current node.
            param4: (list), list of ratios.
        Returns:
            (list), ratios of zeros and ones.
        """
        self.ratios = ratios

        # Check if node is leaf at current branch.
        if self.isLeaf(data, gt, node):
            # Declare leaf node
            node.isleafnode = True

            # Store node's impurity
            node.nodeclass = self.class_of_node(gt)

            # Store ratio of zeros to ones at this leaf.
            self.ratios.append(len(gt[np.where(gt == 0)]) / len(gt))
        else:
            # Find best split
            split_index, threshold = self.find_best_split(data, gt)

            # Split data using best split, and threshold.
            data_splitted_indices1, data_splitted_indices2, gt_splitted1, gt_splitted2 = self.split_data(data, gt, split_index, threshold)

            # Store node's split index, and threshold.
            node.split_index = split_index
            node.threshold = threshold

            # Check if node has a split index or threshold, if not, declare leaf node.
            if node.split_index is None or node.threshold is None:
                node.isLeafnode = True
                node.nodeclass = self.class_of_node(gt)

                # Store ratio of zeros to ones at this leaf.
                self.ratios.append(len(gt[np.where(gt == 0)]) / len(gt))

            # Find data using data splits indices.
            datasplit1 = data[data_splitted_indices1]
            datasplit2 = data[data_splitted_indices2]

            # Initialize left and right node
            node.right = Node()
            node.left = Node()

            # Add depth of node
            node.right.depth = node.depth + 1
            node.left.depth = node.depth + 1

            # Calculate impurity of node
            node.right.nodeclass = self.class_of_node(gt_splitted1)
            node.left.nodeclass = self.class_of_node(gt_splitted2)

            # Initialize right and left branch of tree, and pass in the respective Node objects.
            right = Tree(node.right)
            left = Tree(node.left)

            # Recursively train model, with left and right branches.
            right.train(datasplit1, gt_splitted1, node.right, ratios)
            left.train(datasplit2, gt_splitted2, node.left, ratios)

        return self.ratios

    def isLeaf(self, data, gt, node):
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
        elif self.impurity(gt) < self.minimum_impurity:
            return True
        elif len(data) < self.min_data_nodes:
            return True
        else:
            return False

    def class_of_node(self, gt):
        """
        Method for deciding wheter node is classified as 0 or 1.

        Args:
            param1: (np.ndarray), ground truth, should be of size gt.size = (n,).
        Returns:
            (int), Either 1 or 0.
        """
        # Check if number of elements in ground truth which are really 1, is larger or smaller than number of 0's
        if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
            return 1
        else:
            return 0

    def find_best_split(self, data, groundtruth):
        """
        Method to find the best split index and threshold value in a trainingset.

        --- Disclaimer ---
        This is the most demanding process of this classifier, since the tree iterate over every row, then every value of each column. Then do it again for every split
        ---            ---

        Args:
            param1: (np.ndarray), data
            param2: (np.ndarray), ground ground truth
        Returns:
            (int, float), index of best split, threshold
        """
        # Array which stores the entropy of each row
        entropy_column = np.zeros_like(data[0])
        thresholds = []

        # Loop over the columns of the data
        for index_outer, column in enumerate(data.T):
            # Same as entropy_column, but for the values in the columns
            entropies = np.zeros_like(column)

            # Elements which are zero and one of column
            column_C0 = column[np.where(groundtruth == 0)]
            column_C1 = column[np.where(groundtruth == 1)]

            # Find min and max value of column
            max_value, min_value = np.max(column), np.min(column)

            # Loop over values of rows
            for index_inner, split in enumerate(column):
                # if a split is the max or min value, the total entropy is automatically set to 1 because all nodes at this point either are smaller or larger than max or min.
                if split == max_value or split == min_value:
                    I_tot = 1
                else:
                    # Probability of being over the border given beloning to either of the classes.
                    p_over_split_belongC0 = len(column_C0[column_C0 > split]) / len(column[column > split])

                    # Probability of being under the border given beloning to eiter of the classes.
                    p_under_split_belongC0 = len(column_C0[column_C0 <= split]) / len(column[column <= split])

                    # Total over the split
                    total_over_split = len(column[column > split]) / len(column)

                    # Total under the split
                    total_under_split = 1 - total_over_split

                    # Calculate total impurity
                    I_tot = total_over_split * self.entropy(p_over_split_belongC0) + total_under_split * self.entropy(p_under_split_belongC0)

                # Appending total entropies to list
                entropies[index_inner] = I_tot

            # Find best split, and threshold of the columns
            best_split_of_column = np.min(entropies)
            best_split_value = column[np.where(entropies == np.min(entropies))[0]]

            # If split has multiple choices, choose the first one
            if not isinstance(best_split_value, float):
                thresholds.append(best_split_value[0])
            else:
                thresholds.append(best_split_value)

            # Appending total entropies to list
            entropy_column[index_outer] = best_split_of_column

        # Do the same as above but for all nodes.
        best_overall_split = np.min(entropy_column)
        best_overall_split_index = np.where(entropy_column == best_overall_split)[0][0]
        thresholds = np.asarray(thresholds)
        best_overall_split_value = thresholds[np.where(entropy_column == best_overall_split)][0]

        return best_overall_split_index, best_overall_split_value

    def impurity(self, labels):
        """
        Method for calculating the impurity of given labels.

        Args:
            param1: (np.ndarray), labels, or ground truth
        Returns:
            (float), impurity
        """
        p0 = len(labels[np.where(labels == 0)]) / len(labels)

        # Because 0 * log( 0 ) =def= 0 check if p = 0 or 1
        if p0 == 0 or p0 == 1:
            I = 0
        else:
            I = - p0 * np.log2(p0) - (1 - p0) * np.log2(1 - p0)
        return I

    def predict(self, data):
        """
        Method for classifying data after Tree has been trained.

        Args:
            param1: (np.ndarray), testing data
        Returns:
            (np.ndarray), predicted 0's and 1's.
        """
        predictions = []
        for row in data:
            # This function takes the Tree object itself in, since this has stored all nodes, leafs, etc.
            prob = self.predict_sample(row, self.Tree)
            predictions.append(prob)
        return np.asarray(predictions)

    def predict_sample(self, row, node):
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
            return node.nodeclass

        # Recursively move down the tree until each leaf nodes prediction has been returned.
        if row[node.split_index] > node.threshold:
            prob = self.predict_sample(row, node.right)
        else:
            prob = self.predict_sample(row, node.left)
        return prob

    @staticmethod
    def split_data(data, groundtruth, index, threshold):
        """
        Method for splitting the data.

        This is a staticmethod, which means it does not have any self object tied to it althoug it is a method within a object.

        Args:
            param1: (np.ndarray), data to be splitted.
            param2: (np.ndarray), ground truth
            param3: (int), index of split
            param4: (float), threshold at split
        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray), split1, split2, split1 ground truth, split2 ground truth.
        """
        index = int(index)

        split1, split2 = [], []
        split1_gt, split2_gt = [], []

        # Loop through column to be splitted
        column_at_index = data.T[index]
        for i, val in enumerate(column_at_index):
            # Check each value against threshold.
            if val > threshold:
                split1.append(i)
                split1_gt.append(groundtruth[i])
            else:
                split2.append(i)
                split2_gt.append(groundtruth[i])

        return np.asarray(split1), np.asarray(split2), np.asarray(split1_gt), np.asarray(split2_gt)

    @staticmethod
    def entropy(p_i):
        """
        Method for calculating the entropy.

        This is a staticmethod, which means it does not have any self object tied to it althoug it is a method within a object.

        Args:
            param1: (float), probability
        Returns:
            (float), entropy
        """
        # Because 0 * log( 0 ) =def= 0 check if p = 0 or 1
        if p_i == 1 or p_i == 0:
            return 0
        else:
            return - p_i * np.log2(p_i) - (1 - p_i) * np.log2(1 - p_i)

    def performance(self, classification, groundTruth):
        """
        Method for calculating performance measures of classifiers

        Args:
            param1: (np.ndarray), classified 0's and 1's.
            param2: (np.ndarray), actual 0's and 1's.
        Returns:
            (np.ndarray, float), confusion matrix, accuracy of classifier.
        """
        # Gives index of which elements are 0 and 1.
        classified0 = np.where(classification == 0)
        classified1 = np.where(classification == 1)
        actual0 = np.where(groundTruth == 0)
        actual1 = np.where(groundTruth == 1)

        # Send theese to the confusionMatrix method
        confmat = self.confusionMatrix(classified0, classified1, actual0, actual1)
        accuracy = (confmat[0][0] + confmat[1][1]) / np.sum(confmat).astype("float32")
        return confmat, accuracy

    def confusionMatrix(self, classified0, classified1, actual0, actual1):
        # Note: this method has been copied from my own work in Portfolio Assignment 1 and Problem 1.
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

    # -------- (2c) --------
    def ROC(self, ratios, gt, n=101):
        """
        Method for plotting the ROC curve and calculating AUC score of this classifier.

        Args:
            param1: (np.ndarray), list of ratios fund by dividing ones by total rows in the fit method.
            param2: (np.ndarray), ground truth
            param3: (int), optional, standard as n = 101, number of steps.
        """
        # Array of boundaries
        boundaries = np.linspace(0, 1, n)
        tprates = []
        fprates = []

        for boundary in boundaries:

            # If ratios > boundary, return 1, else return 0 is what the line under does.
            classification = np.where(ratios > boundary, 1, 0)

            # Only interested in the confusion matrix
            confmat, _ = self.performance(classification, gt)
            tp = confmat[0][0]
            fn = confmat[0][1]
            fp = confmat[1][0]
            tn = confmat[1][1]

            # Calculate and append tp- and fprates.
            tprates.append(tp / (tp + fn))
            fprates.append(fp / (fp + tn))

        tprates = np.asarray(tprates)
        fprates = np.asarray(fprates)

        # my main plot with false postive rates on the first axis, and true positive rates on the second axis.
        mainplot = plt.plot(fprates, tprates, label="Decision Tree Classifier")

        # The diagonal line, generally known as the x = y - line
        xyLine = plt.plot(np.linspace(np.min(fprates), np.max(fprates), len(boundaries)), np.linspace(np.min(tprates), np.max(tprates), len(boundaries)), "--", label="x = y")

        # Using sklearn.metrics AUC method
        AUC = metrics.auc(fprates, tprates)

        # Misc for naming the plots.
        title = plt.title("ROC curve for Decision Tree")
        xlab = plt.xlabel("False positive rates")
        ylab = plt.ylabel("True positive rates")
        plt.legend(title="n = {} Step size of thresholds\nAUC = {:.3f}".format(n, AUC))
        plt.show()

def main():
    seals = Tree()
    ratios = seals.fit(traindata, ground_truth)
    classified = seals.predict(testdata)
    confusionmat, accuracy = seals.performance(classified, test_gt)
    print(confusionmat, accuracy)
    seals.ROC(ratios, test_gt)

if __name__ == "__main__":
    main()
