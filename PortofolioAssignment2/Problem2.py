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
    def __init__(self, list_of_split_index=list(), list_of_threshold=list(), node_list=list(), depth_list=list()):
        self.minimum_impurity = 0.6
        self.min_data_nodes = 40
        self.max_recursion_depth = 5
        self.list_of_split_index = list_of_split_index
        self.list_of_threshold = list_of_threshold
        self.node_list = node_list
        self.depth_list = depth_list

    def fit(self, data, gt, split_index=None, threshold=None, depth=0, node="root"):
        self.depth = depth
        if self.depth < self.max_recursion_depth:
            if self.impurity(gt) < self.minimum_impurity:
                if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                    print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)

                else:
                    print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
            else:
                if len(data) < self.min_data_nodes:
                    if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                        print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
                    else:
                        print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
                else:
                    # Find best split
                    split_index, threshold = self.find_best_split(data, gt)
                    data_splitted1, data_splitted2, gt_splitted1, gt_splitted2 = self.split_data(data, gt, split_index, threshold, node)

                    # self.list_of_node.append(node + " " + str(depth))
                    self.node_list.append(node)
                    self.depth_list.append(depth)

                    datasplit1 = data[data_splitted1]
                    datasplit2 = data[data_splitted2]
                    # print("right split: {}, left split: {}".format(datasplit1.shape, datasplit2.shape))

                    self.list_of_split_index.append(split_index)
                    self.list_of_threshold.append(threshold)


                    right = Tree(self.list_of_split_index, self.list_of_threshold)
                    left = Tree(self.list_of_split_index, self.list_of_threshold)

                    right.fit(datasplit1, gt_splitted1, split_index, threshold, self.depth + 1, "right")
                    left.fit(datasplit2, gt_splitted2, split_index, threshold, self.depth + 1, "left")

                    # right.fit(data_splitted1, gt_splitted1, split_index, threshold, self.depth + 1)
                    # left.fit(data_splitted2, gt_splitted2, split_index, threshold, self.depth + 1)

                    return self.list_of_split_index, self.list_of_threshold, self.node_list, self.depth_list
        else:
            if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
            else:
                print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)

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
    def split_data(data, groundtruth, index, threshold, node):
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
    def predictdunno(data, gt, split_ind, thresholds, nodes, depths, indices, depth=0, leaf=list()):
        branch1, branch2 = [], []
        branch1gt, branch2gt = [], []
        indices1, indices2 = [], []

        for index, row in enumerate(data):
            if row[split_ind[depth]] > thresholds[depth]:
                branch1.append(row)
                branch1gt.append(gt[index])
                indices1.append(index)

            else:
                branch2.append(row)
                branch2gt.append(gt[index])
                indices2.append(index)

        branch1indices = np.zeros_like(indices1)
        branch2indices = np.zeros_like(indices2)

        for index, i in enumerate(indices1):
            branch1indices[index] = indices[i]
        for index, i in enumerate(indices2):
            branch2indices[index] = indices[i]

        if dep < np.max(depths):
            if nodes[dep] == 'right':
                print("right")
                Tree.predictdunno(branch1, branch1gt, split_ind, thresholds, nodes, depths, branch1indices, dep)

            if nodes[dep] == 'left':
                print("left")
                Tree.predictdunno(branch2, branch2gt, split_ind, thresholds, nodes, depths, branch2indices, dep)
        return leaf

    @staticmethod
    def predict(data, gt, split_ind, thresholds, nodes, depths, indices, depth=0, leaf=list()):
        # loope gjennom nodes, sjekke om det er en leaf. hvis det er leaf, classifiser som leaf, hvis ikke, split data.

        for index_of_node, node in enumerate(nodes):
            rightdata, leftdata = [], []
            rightgt, leftgt = [], []
            for i, row in enumerate(data):
                if row[split_ind[depth]] > thresholds[depth]:
                    rightdata.append(row)
                    rightgt.append(gt[i])
                else:
                    leftdata.append(row)
                    leftgt.append(gt[i])

            depth = depths[index_of_node]
            if node == "right":
                print("right at depth %s, number of rows %d" % (depth, len(rightdata)))
                Tree.predict(rightdata, rightgt, split_ind, thresholds, depths, indices, depth)
            if node == "left":
                print("left at depth %s, number of rows %d" % (depth, len(rightdata)))
                Tree.predict(leftdata, leftgt, split_ind, thresholds, depths, indices, depth)


    @staticmethod
    def entropy(p_i):
        if p_i == 1 or p_i == 0:
            return 0
        else:
            return - p_i * np.log2(p_i) - (1 - p_i) * np.log2(1 - p_i)

def main():
    # seals = Tree()
    # split_index, thresholds, node, depth = seals.fit(traindata, ground_truth)
    # print(split_index, thresholds, node, depth)

    data1 = [[0, 1, 1, 106, 0, 11, 1, 0, 103], [-0.8347, 2.8826, 7.7557, -3.4148, 2.6479, 2.9597, -3.6158, -9.6394, 1.5368], ['root', 'right', 'right', 'right', 'left', 'right', 'left', 'left', 'right'], [0, 1, 2, 3, 3, 4, 1, 2, 3]]
    indexlist = np.array(range(0, len(testdata)))

    Tree.predict(testdata, test_gt, data1[0], data1[1], data1[2], data1[3], indexlist)

if __name__ == "__main__":
    main()
