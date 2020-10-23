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

test_gt = testdata[:, 0]
testdata = np.delete(testdata, 0, axis=1)

class Tree:
    """
    Class for decision tree.

    Tree takes no arguements.
    """
    def __init__(self, list_of_split_index=list(), list_of_threshold=list()):
        self.minimum_impurity = 0.6
        self.min_data_nodes = 40
        self.max_recursion_depth = 5
        self.list_of_split_index = list_of_split_index
        self.list_of_threshold = list_of_threshold

    def fit(self, data, gt, split_index=None, threshold=None, depth=0, ):
        self.depth = depth
        if self.depth < self.max_recursion_depth:
            if self.impurity(gt) < self.minimum_impurity:
                if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                    print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth))
                else:
                    print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth))
            else:
                if len(data) < self.min_data_nodes:
                    if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                        print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth))
                    else:
                        print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth))
                else:
                    # Find best split
                    split_index, threshold = self.find_best_split(data, gt)
                    data_splitted1, data_splitted2, gt_splitted1, gt_splitted2 = self.split(data, gt, split_index, threshold)

                    datasplit1 = data[data_splitted1]
                    datasplit2 = data[data_splitted2]
                    print(datasplit1.shape)

                    self.list_of_split_index.append(split_index)
                    self.list_of_threshold.append(threshold)


                    right = Tree(self.list_of_split_index, self.list_of_threshold)
                    left = Tree(self.list_of_split_index, self.list_of_threshold)

                    right.fit(datasplit1, gt_splitted1, split_index, threshold, self.depth + 1)
                    left.fit(datasplit2, gt_splitted2, split_index, threshold, self.depth + 1)
                    # right.fit(data_splitted1, gt_splitted1, split_index, threshold, self.depth + 1)
                    # left.fit(data_splitted2, gt_splitted2, split_index, threshold, self.depth + 1)

                    return self.list_of_split_index, self.list_of_threshold
        else:
            if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth))
            else:
                print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth))

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

    @staticmethod
    def split(data, groundtruth, index, threshold):
        """
        Method for splitting the data!
        """
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
    def predict(data, gt, split_ind, thresholds):

        split = split(data, gt, index, threshold)


    @staticmethod
    def entropy(p_i):
        if p_i == 1 or p_i == 0:
            return 0
        else:
            return - p_i * np.log2(p_i) - (1 - p_i) * np.log2(1 - p_i)

def main():
    seals = Tree()
    split_index, thresholds = seals.fit(traindata, ground_truth)
    print(split_index, thresholds)
    # print(Tree.split(traindata, ground_truth, 0, -1))

    # predict(testdata, test_gt, split_index, thresholds)

if __name__ == "__main__":
    main()
