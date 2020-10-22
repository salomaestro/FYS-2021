import numpy as np
import matplotlib.pyplot as plt
import os
# from sklearn import metrics

# Import and initialize the data
dirname = os.path.dirname(__file__)
filename_seals_train = os.path.join(dirname, "seals_train.csv")
filename_seals_test = os.path.join(dirname, "seals_test.csv")
filename_seals_images_test = os.path.join(dirname, "seals_images_test.csv")
traindata = np.genfromtxt(filename_seals_train, delimiter=" ")
testdata = np.genfromtxt(filename_seals_test, delimiter=" ")

ground_truth = traindata[:, 0]
traindata = np.delete(traindata, 0, axis=1)

class Tree:
    """
    Class for decision tree.

    Tree takes no arguements.
    """
    def __init__(self):
        self.minimum_impurity = 0.8
        self.min_data_nodes = 40
        self.max_recursion_depth = 6

    def fit(self, data, gt, depth, node="Root", direction=0):
        self.depth = depth
        self.node = node
        if self.depth < self.max_recursion_depth:
            if self.impurity(gt) < self.minimum_impurity:
                if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                    print("Leaf node with {} datapoints, belongs to class 1, at depth = {}".format(len(gt), self.depth))
                else:
                    print("Leaf node with {} datapoints, belongs to class 0, at depth = {}".format(len(gt), self.depth))
            else:
                if len(data) < self.min_data_nodes:
                    if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                        print("Leaf node with shape {}, belongs to class 1, at depth = {}".format(len(gt), self.depth))
                    else:
                        print("Leaf node with shape {}, belongs to class 0, at depth = {}".format(len(gt), self.depth))
                else:
                    # Find best split
                    split_index, threshold = self.find_best_split(data, gt)
                    data_splitted1, data_splitted2, gt_splitted1, gt_splitted2 = self.split(data, gt, split_index, threshold)

                    datasplit1 = data[data_splitted1]
                    datasplit2 = data[data_splitted2]

                    right = Tree()
                    left = Tree()
                    
                    print(self.node)

                    right.fit(datasplit1, gt_splitted1, self.depth + 1, {"Right rec_depth = {}".format(self.depth):"Threshold: {:.3f}".format(threshold)})
                    left.fit(datasplit2, gt_splitted2, self.depth + 1, {"Left rec_depth = {}".format(self.depth):"Threshold: {:.3f}".format(threshold)})
        else:
            print("max recursion depth exceeded, remainder is classified by majority in class.")
            if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                print("Leaf node with shape {}, belongs to class 1, at depth = {}".format(len(gt), self.depth))
            else:
                print("Leaf node with shape {}, belongs to class 0, at depth = {}".format(len(gt), self.depth))

    @staticmethod
    def split(data, groundtruth, index, threshold):
        """
        Method for splitting the data!
        """
        split1, split2 = [], []
        split1_gt, split2_gt = [], []
        for i, val in enumerate(data.T[index]):
            if val > threshold:
                split1.append(i)
                split1_gt.append(groundtruth[i])
            if val <= threshold:
                split2.append(i)
                split2_gt.append(groundtruth[i])

        return np.asarray(split1), np.asarray(split2), np.asarray(split1_gt), np.asarray(split2_gt)

    def find_best_split(self, data, groundtruth):
        entropy_column = np.zeros_like(data[0])
        # thresholds = np.zeros_like(groundtruth)
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

                    total_under_split = 1 - total_over_split

                    I_tot = total_over_split * self.entropy(p_over_split_belongC0) + total_under_split * self.entropy(p_under_split_belongC0)
                entropies[index_inner] = I_tot

            best_split_of_column = np.min(entropies)
            best_split_value = column[np.where(entropies == np.min(entropies))[0]]

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
        pass

    @staticmethod
    def entropy(p_i):
        if p_i == 1 or p_i == 0:
            return 0
        else:
            return - p_i * np.log2(p_i) - (1 - p_i) * np.log2(1 - p_i)

def main():
    seals = Tree()
    seals.fit(traindata, ground_truth, 1)

if __name__ == "__main__":
    main()
