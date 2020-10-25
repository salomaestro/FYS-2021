import matplotlib.pyplot as plt
import numpy as np
import os

directory = os.path.dirname(__file__)


sealsfile = os.path.join(directory, 'seals_test.csv')
sealstraining = os.path.join(directory, 'seals_train.csv')

sealstest = np.genfromtxt(sealsfile, delimiter=' ')
sealstraining = np.genfromtxt(sealstraining, delimiter=' ')

sealstesttruth = sealstest.T[0]
sealstest = np.delete(sealstest, 0, axis=1)

sealstrainingtruth = sealstraining.T[0]
sealstraining = np.delete(sealstraining, 0, axis=1)



class Tree:

    def __init__(self, data, TV=list(), IV=list(), IDnodes=list(), depthlist=list(), leaf=list(), leafdir=list()):
        self.thedataset = data
        self.minimpurity = 0.76
        self.thresholdvalue = TV
        self.thresholdindex = IV
        self.IDnodes = IDnodes
        self.depthlist = depthlist
        self.leaf = leaf
        self.leafdirection = leafdir


    def entropy(self, p):
        if p == 1 or p == 0:
            return 0
        else:
            I = -p*np.log2(p) - (1 - p)*np.log2(1 - p)

            return I

    def findthreshold(self, dataset, truth):
        data = dataset.T
        entropypercolumn = []
        thresholds = []
        for w, column in enumerate(data):
            entropies = []
            N = np.shape(column)[0]
            for index, value in enumerate(column):
                above_value = np.count_nonzero(truth[np.where(column > value)[0]])
                totalabove_value = np.shape(np.where(column > value)[0])[0]
                below_value = np.count_nonzero(truth[np.where(column <= value)[0]])
                totalbelow_value = np.shape(np.where(column <= value)[0])[0]

                if totalbelow_value == 0 or totalabove_value == 0:
                    I_tot = 1
                elif above_value/totalabove_value == 0 or above_value/totalabove_value == 1 or below_value/totalbelow_value == 0 or below_value/totalbelow_value == 1:
                    I_tot = 1
                else:
                    I_tot = (above_value/N)*self.entropy(above_value/totalabove_value) + (below_value/N)*self.entropy(below_value/totalbelow_value)
                entropies.append(I_tot)
            entropypercolumn.append(np.min(entropies))
            thresholds.append(column[np.where(entropies == np.min(entropies))[0]][0])
        split_index = np.where(entropypercolumn == np.min(entropypercolumn))[0][0]
        split = thresholds[split_index]
        return split, split_index

    def impurity(self, data, truth):
        truthlist = []
        for k in range(0, np.shape(data)[0]):
            truthlist.append(truth[k])
        ones = np.count_nonzero(truthlist)/(np.shape(data)[0])
        imp = self.entropy(ones)
        return imp

    def splitdata(self, data, truth, threshold):
        self.thresholdvalue.append(threshold[0])
        self.thresholdindex.append(threshold[1])
        split1 = []
        split2 = []
        split1truths = []
        split2truths = []
        datacol = data.T[threshold[1]]
        for index, value in enumerate(datacol):
            if value > threshold[0]:
                split1.append(self.thedataset[index])
                split1truths.append(truth[index])
            elif value <= threshold[0]:
                split2.append(self.thedataset[index])
                split2truths.append(truth[index])
        return np.array(split1), np.array(split2), np.array(split1truths), np.array(split2truths)

    def fit(self, depth, data, truth, node):
        self.depth = depth
        if self.impurity(data, truth) < self.minimpurity:
            if np.count_nonzero(truth) > np.shape(truth)[0] - np.count_nonzero(truth):
                print('Leaf node with', np.shape(truth)[0], 'datapoints classified as 1, with depth', self.depth)
                self.depthlist.append(self.depth)
                self.leafdirection.append(1)

            else:
                print('Leaf node with', np.shape(truth)[0], 'datapoints classified as 0, with depth', self.depth)
                self.depthlist.append(self.depth)
                self.leafdirection.append(0)

        else:

            thresholdinfo = self.findthreshold(data, truth)
            newdata = self.splitdata(data, truth, thresholdinfo)
            self.IDnodes.append('branch-%s' % node)
            branch1 = Tree(sealstraining)
            branch2 = Tree(sealstraining)
            if len(newdata[0]) == 0:
                pass
            else:
                branch1.fit(self.depth + 1, newdata[0], newdata[2], '1')
            if len(newdata[1]) == 0:
                pass
            else:
                branch2.fit(self.depth + 1, newdata[1], newdata[3], '2')
            return self.thresholdvalue, self.thresholdindex, self.IDnodes, self.depthlist, self.leafdirection


    def classifydata(self, data, truth, depth, thresholds, thresholdindices, IDnodes, maxdepth, indices):
        branch1 = []
        branch1truth = []
        indices1 = []
        branch2 = []
        branch2truth = []
        indices2 = []
        for index, row in enumerate(data):
            if row[thresholdindices[depth]] > thresholds[depth]:
                branch1.append(row)
                branch1truth.append(truth[index])
                indices1.append(index)
            else:
                branch2.append(row)
                branch2truth.append(truth[index])
                indices2.append(index)

        branch1indices = np.array(range(0, len(branch1)))
        branch2indices = np.array(range(0, len(branch2)))
        for index, i in enumerate(indices1):
            branch1indices[index] = indices[i]
        for index, i in enumerate(indices2):
            branch2indices[index] = indices[i]

        depth += 1
        if depth < max(maxdepth):
            if IDnodes[depth] == 'branch-1':
                self.classifydata(branch1, branch1truth, depth, thresholds, thresholdindices, IDnodes, maxdepth, branch1indices)
                self.leaf.append(branch2indices)
            elif IDnodes[depth] == 'branch-2':
                self.classifydata(branch2, branch2truth, depth, thresholds, thresholdindices, IDnodes, maxdepth, branch2indices)
                self.leaf.append(branch1indices)
        else:
            self.leaf.append(branch1indices)
            self.leaf.append(branch2indices)
        return self.leaf


    def checkclassification(self, leaves, leaftruths, depthlist):
        depthlist = np.flip(np.argsort(depthlist))
        sortedtruths = np.zeros_like(leaftruths)
        for index, i in enumerate(depthlist):
            sortedtruths[index] = leaftruths[i]


        class0cluster = []
        class1cluster = []
        for index, i in enumerate(sortedtruths):
            if i == 1:
                class1cluster.append(leaves[index])
            else:
                class0cluster.append(leaves[index])
        class0 = []
        class1 = []
        for k in class0cluster:
            for r in k:
                class0.append(r)
        for j in class1cluster:
            for b in j:
                class1.append(b)
        classed = np.zeros_like(sealstesttruth)
        for i in class1:
            classed[i] = 1

        return classed

    def confusion(self, sortedclass, trueclass):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0, np.shape(trueclass)[0]):
            if sortedclass[i] == 0 and trueclass[i] == 0:
                tp += 1
            elif sortedclass[i] == 0 and trueclass[i] == 1:
                fp += 1
            elif sortedclass[i] == 1 and trueclass[i] == 0:
                fn += 1
            elif sortedclass[i] == 1 and trueclass[i] == 1:
                tn += 1

        matrix = np.array([[tp, fn], [fp, tn]])
        accuracy = 1 - (fp + fn)/(tp + fp + fn + tn)
        return matrix, accuracy

tree = Tree(sealstraining)

info = tree.fit(0, sealstraining, sealstrainingtruth, 'root')
indexlist = np.array(range(0, len(sealstest)))
placedinleafnodes = tree.classifydata(sealstest, sealstesttruth, 0, info[0], info[1], info[2], info[3], indexlist)
classified = tree.checkclassification(placedinleafnodes, info[4], info[3])
confusionmatrix = tree.confusion(classified, sealstesttruth)
print(confusionmatrix[0])
print('Where the classifiers accuracy is:', confusionmatrix[1])
