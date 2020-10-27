import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
filename_seals_test = os.path.join(dirname, "seals_test.csv")
filename_seals_images_test = os.path.join(dirname, "seals_images_test.csv")
traindata = np.genfromtxt(filename_seals_train, delimiter=" ")
testdata = np.genfromtxt(filename_seals_test, delimiter=" ")

ground_truth = traindata[:, 0]
traindata = np.delete(traindata, 0, axis=1)

test_gt = testdata[:, 0]
testdata = np.delete(testdata, 0, axis=1)

class Node:
    def __init__(self):
        # left and right nodes
        self.right = None
        self.left = None

        # Assigned when find_best_split function is called
        self.split_index = None
        self.threshold = None

        # probability for object inside the Node to belong for each of the given classes
        self.impurity = None

        # depth of the given node
        self.depth = None

        # Is declared leaf if True
        self.isLeafnode = False

        self.ratio = None

class Tree:
    """
    Class for decision tree.

    Tree takes no arguements.
    """
    def __init__(self, node=None, minimum_impurity=0.6, min_data_nodes=40, max_recursion_depth=5):
        self.minimum_impurity = minimum_impurity
        self.min_data_nodes = min_data_nodes
        self.max_recursion_depth = max_recursion_depth
        self.Tree = node

    def fit(self, data, gt):
        self.Tree = Node()
        self.Tree.depth = 1
        self.Tree.impurity = self.probability_on_node(gt)

        self.train(data, gt, self.Tree)

    def train(self, data, gt, node, ratios=list()):
        self.ratios = ratios
        if self.isLeaf(data, gt, node):
            node.isleafnode = True
            node.impurity = self.probability_on_node(gt)
            self.ratios.append(len(np.where(gt == 1)) / len(gt))
        else:
            # Find best split
            split_index, threshold = self.find_best_split(data, gt)
            data_splitted_indices1, data_splitted_indices2, gt_splitted1, gt_splitted2 = self.split_data(data, gt, split_index, threshold)

            node.split_index = split_index
            node.threshold = threshold

            if node.split_index is None or node.threshold is None:
                node.isLeafnode = True
                node.impurity = self.probability_on_node(gt)
                self.ratios.append(len(np.where(gt == 1)) / len(gt))


            datasplit1 = data[data_splitted_indices1]
            datasplit2 = data[data_splitted_indices2]

            node.right = Node()
            node.left = Node()

            node.right.depth = node.depth + 1
            node.left.depth = node.depth + 1

            node.right.impurity = self.probability_on_node(gt_splitted1)
            node.left.impurity = self.probability_on_node(gt_splitted2)

            right = Tree(node.right)
            left = Tree(node.left)

            right.train(datasplit1, gt_splitted1, node.right, ratios)
            left.train(datasplit2, gt_splitted2, node.left, ratios)
        return self.ratios

    def isLeaf(self, data, gt, node):
        if node.depth >= self.max_recursion_depth:
            return True
        elif self.impurity(gt) < self.minimum_impurity:
            return True
        elif len(data) < self.min_data_nodes:
            return True
        else:
            return False

    def probability_on_node(self, gt):
        if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
            return 1
        else:
            return 0

    def find_best_split(self, data, groundtruth):
        entropy_column = np.zeros_like(data[0])
        thresholds = []

        for index_outer, column in enumerate(data.T):
            entropies = np.zeros_like(column)

            column_C0 = column[np.where(groundtruth == 0)]
            column_C1 = column[np.where(groundtruth == 1)]

            max_value, min_value = np.max(column), np.min(column)

            for index_inner, split in enumerate(column):
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

                    I_tot = total_over_split * self.entropy(p_over_split_belongC0) + total_under_split * self.entropy(p_under_split_belongC0)
                entropies[index_inner] = I_tot

            best_split_of_column = np.min(entropies)
            best_split_value = column[np.where(entropies == np.min(entropies))[0]]

            # If split has multiple choices, choose the first one
            if not isinstance(best_split_value, float):
                thresholds.append(best_split_value[0])
            else:
                thresholds.append(best_split_value)

            entropy_column[index_outer] = best_split_of_column

        best_overall_split = np.min(entropy_column)
        best_overall_split_index = np.where(entropy_column == best_overall_split)[0][0]
        thresholds = np.asarray(thresholds)
        best_overall_split_value = thresholds[np.where(entropy_column == best_overall_split)][0]

        return best_overall_split_index, best_overall_split_value

    def impurity(self, labels):
        p0 = len(labels[np.where(labels == 0)]) / len(labels)
        if p0 == 0 or p0 == 1:
            I = 1
        else:
            I = - p0 * np.log2(p0) - (1 - p0) * np.log2(1 - p0)
        return I

    def predict(self, data):
        predictions = []
        for row in data:
            prob = self.predict_sample(row, self.Tree)
            predictions.append(prob)
        return np.asarray(predictions)

    def predict_sample(self, row, node):
        if node.isLeafnode or node.threshold is None:
            return node.impurity

        if row[node.split_index] > node.threshold:
            prob = self.predict_sample(row, node.right)
        else:
            prob = self.predict_sample(row, node.left)
        return prob

    @staticmethod
    def split_data(data, groundtruth, index, threshold):
        """
        Method for splitting the data!
        """
        index = int(index)

        split1, split2 = [], []
        split1_gt, split2_gt = [], []

        # Loop through column to be splitted
        column_at_index = data.T[index]
        for i, val in enumerate(column_at_index):
            if val > threshold:
                split1.append(i)
                split1_gt.append(groundtruth[i])
            else:
                split2.append(i)
                split2_gt.append(groundtruth[i])

        return np.asarray(split1), np.asarray(split2), np.asarray(split1_gt), np.asarray(split2_gt)

    @staticmethod
    def entropy(p_i):
        if p_i == 1 or p_i == 0:
            return 0
        else:
            return - p_i * np.log2(p_i) - (1 - p_i) * np.log2(1 - p_i)

    def PrecisionMethod(self, classification, groundTruth):
        classified0 = np.where(classification == 0)
        classified1 = np.where(classification == 1)
        actual0 = np.where(groundTruth == 0)
        actual1 = np.where(groundTruth == 1)
        confmat = self.confusionMatrix(classified0, classified1, actual0, actual1)
        accuracy = (confmat[0][0] + confmat[1][1]) / np.sum(confmat).astype("float32")
        return confmat, accuracy

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

    def ROC(self):
        pass

def main():
    seals = Tree()
    print(seals.fit(traindata, ground_truth))
    print("training complete")
    classified = seals.predict(testdata)
    confusionmat = seals.PrecisionMethod(classified, test_gt)
    print(confusionmat)

if __name__ == "__main__":
    main()
