def fit(self, data, gt, split_index=None, threshold=None, depth=0, node="root"):
    self.depth = depth
    if self.depth < self.max_recursion_depth:
        if self.impurity(gt) < self.minimum_impurity:
            if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
            else:
                print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
            self.node_list.append(node)
            self.depth_list.append(depth)
            self.leaf_list.append(1)
        else:
            if len(data) < self.min_data_nodes:
                if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
                    print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
                else:
                    print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
                self.node_list.append(node)
                self.depth_list.append(depth)
                self.leaf_list.append(1)
            else:
                # Find best split
                split_index, threshold = self.find_best_split(data, gt)
                data_splitted1, data_splitted2, gt_splitted1, gt_splitted2 = self.split_data(data, gt, split_index, threshold, node)

                # self.list_of_node.append(node + " " + str(depth))
                self.node_list.append(node)
                self.depth_list.append(depth)
                self.leaf_list.append(0)

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

                return self.list_of_split_index, self.list_of_threshold, self.node_list, self.depth_list, self.leaf_list
    else:
        if len(gt[np.where(gt == 1)]) > len(gt[np.where(gt == 0)]):
            print("Became leaf node after split index {}, with threshold {}, belongs to class 1, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
        else:
            print("Became leaf node after split index {}, with threshold {}, belongs to class 0, at depth = {}".format(split_index, threshold, self.depth), len(gt), node)
        self.node_list.append(node)
        self.depth_list.append(depth)
        self.leaf_list.append(1)
