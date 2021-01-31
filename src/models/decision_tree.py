import numpy as np
import math
from models.tree_node import TreeNode


class DecisionTree:

    def __init__(self, dataset, target, max_depth, min_leaf):
        self.dataset = dataset
        self.target = target
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def create_tree(self, dataset_index, vocabulary_index, m=0, depth=0):
        if dataset_index.size <= self.min_leaf:
            node = TreeNode(None, depth + 1, m, True)
            return node
        if self.all_same(dataset_index):
            node = TreeNode(None, depth + 1, self.target[dataset_index[0]], True)
            return node
        elif vocabulary_index.size == 2 or depth >= self.max_depth:
            node = TreeNode(None, depth + 1, self.majority_value(dataset_index), True)
            return node
        else:
            best_split = self.choose_attribute(dataset_index, vocabulary_index)
            m = self.majority_value(dataset_index)
            node = TreeNode(best_split, depth + 1, m)
            for cl in range(2):
                new_dataset = self.split(dataset_index, best_split, cl)
                branch = self.create_tree(new_dataset, np.delete(vocabulary_index, best_split, axis=0), m, depth + 1)

                if cl:
                    node.left = branch
                else:
                    node.right = branch

            return node

    def all_same(self, dataset_index):
        """

        Args:
            dataset_index: A list that contains the index of our current dataset to the starting dataset.

        Returns: True if all currents dataset's targets have same value or false if they have not same value.

        """
        return all(self.target[x] == self.target[dataset_index[0]] for x in dataset_index)

    def majority_value(self, dataset_index):
        """

        Args:
            dataset_index: A list that contains the index of our current dataset to the starting dataset.

        Returns: 1 if the majority of current's dataset targets are of class 1 or 0 if not.

        """
        if np.count_nonzero(self.target[dataset_index]) > dataset_index.size / 2:
            return 1
        else:
            return 0

    def information_gain(self, dataset_index, attr):
        """ Computes information gain for a given term.
        Args:
            dataset_index: A list that contains the index of our current dataset to the starting dataset.
            attr: The position of the term that we want to calculate information gain.

        Returns:
            The information gain of the given term as a float.
        """
        target_count = np.zeros(2)
        target_count[1] = np.count_nonzero(self.target[dataset_index])
        target_count[0] = dataset_index.size - target_count[1]

        entropy = 0
        for p in range(len(set(self.target))):
            try:
                entropy -= target_count[p] / dataset_index.size * math.log2(target_count[p] / dataset_index.size)
            except ValueError:
                # when x in log(x) is zero, add 0 to mutual information
                entropy -= 0

        count = self.count_values(dataset_index, attr)

        ig = 0
        for row in range(count.shape[0]):
            x = float(np.sum(count[row][:]))
            for col in range(count.shape[1]):
                if count[row][col] > 0 and x > 0:
                    ig += (x / dataset_index.size) * \
                          (-((count[row][col] / x) * math.log2(count[row][col] / x)))
                else:
                    # when x in log(x) is zero, add 0 to mutual information
                    ig += 0

        ig = entropy - ig

        return ig

    def split(self, dataset_index, best_split, cl):
        """
        Creates a new dataset with tuples which at best_split position have value cl.

        Args:
            dataset_index: A list that contains the index of our current dataset to the starting dataset.
            best_split: The index of the attribute with the maximum information gain.
            cl: The value tuples we want have at best split.

        Returns: A new list that contains the index of our current dataset to the starting dataset.

        """

        data = []
        for txt in dataset_index:
            if self.dataset[txt][best_split] == cl:
                data.append(txt)
        dataset = np.array(data)

        return dataset

    def count_values(self, dataset_index, attr):
        """
        Calculates the number of classes 0 and 1 where dataset's value at attr position is 1 and 0 .

        Args:
            dataset_index: A list that contains the index of our current dataset to the starting dataset.
            attr: the dataset index on which we want to calculate the values.

        Returns: A 2*2 array that contains all sums.

        """
        count = np.zeros((2, len(set(self.target))))

        for txt in dataset_index:
            if self.dataset[txt][attr]:
                if self.target[txt]:
                    count[1][1] += 1
                else:
                    count[1][0] += 1
            else:
                if self.target[txt]:
                    count[0][1] += 1
                else:
                    count[0][0] += 1

        return count

    def choose_attribute(self, dataset_index, vocabulary_index):
        """
        Finds the attribute with the maximum information gain.

        Returns:

            pos: the position of this attribute
        """
        maximum = 0
        pos = 0

        for attr in range(vocabulary_index.size):
            ig = self.information_gain(dataset_index, vocabulary_index[attr])
            if ig > maximum:
                maximum = ig
                pos = attr
        return pos
