import numpy as np
import math
from RandomForest.TreeNode import TreeNode


class Tree:

    def __init__(self, max_depth, min_leaf):
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def create_tree(self, dataset, target, depth=0):
        if len(dataset) == 0:
            return None
        if all(x == target[0] for x in target):
            node = TreeNode(None, depth + 1, target[0], True)
            return node
        elif len(dataset[0]) == 0 or depth >= self.max_depth-1:
            if np.count_nonzero(target) > len(target)/2:
                cl = 1
            else:
                cl = 0
            node = TreeNode(None, depth + 1, cl, True)
            return node
        else:
            best_split = choose_attribute(dataset, target)
            left_dataset, left_target, right_dataset, right_target = split(dataset, target, best_split)
            if len(left_dataset) < self.min_leaf or len(right_dataset) < self.min_leaf:
                if np.count_nonzero(target) > len(target) / 2:
                    cl = 1
                else:
                    cl = 0
                node = TreeNode(None, depth + 1, cl, True)
            else:
                node = TreeNode(best_split, depth + 1)
                print("left", left_dataset, "c_left", left_target)
                print("right", right_dataset, "c_right", right_target)
                node.left = self.create_tree(left_dataset, left_target, depth + 1)
                node.right = self.create_tree(right_dataset, right_target, depth + 1)
            return node


def split(dataset, target, best_split):
    """
    Splits the dataset and the class list in two smaller datasets and lists. Also deletes the best attribute
    column, so that cannot be used again.

    Returns:
        data_left: Left's node dataset
        c_left: Left's dataset class list
        data_right: Right's node dataset
         c_left: Right's dataset class list
    """

    data_left = []
    c_left = []
    data_right = []
    c_right = []
    for txt in range(len(dataset)):
        if dataset[txt][best_split]:
            data_left.append(dataset[txt])
            c_left.append(target[txt])
        else:
            data_right.append(dataset[txt])
            c_right.append(target[txt])
    if len(dataset) > 0:
        data_left = np.delete(data_left, best_split, axis=1)
        data_right = np.delete(data_right, best_split, axis=1)
    return data_left, c_left, data_right, c_right


def choose_attribute(dataset, target):
    """
    Finds the attribute with the maximum information gain.

    Returns:

        pos: the position of this attribute
    """
    maximum = 0
    pos = 0

    for attr in range(len(dataset[0])):
        ig = information_gain(dataset, target, attr)
        if ig > maximum:
            maximum = ig
            pos = attr
    return pos


def entropy(target):
    """
    Calculates H(c) entropy

    Returns: value of H(c)

    """
    c1 = np.count_nonzero(target)
    c0 = len(target) - c1
    return calculate_entropy(float(c1) / len(target), float(c0) / len(target))


def information_gain(dataset, target, y):
    """
    Calculates information gain value of one column.

    Args:
        target:
        dataset:
        y: value of the column

    Returns: the information gain value

    """
    x1, x0, x1c1, x1c0, x0c1, x0c0 = count_values(dataset, target, y)
    if x1:
        px1c1 = float(x1c1) / x1
        px1c0 = float(x1c0) / x1
    else:
        px1c1 = 0
        px1c0 = 0
    if x0:
        px0c1 = float(x0c1) / x0
        px0c0 = float(x0c0) / x0
    else:
        px0c1 = 0
        px0c0 = 0
    px1 = float(x1) / len(dataset)
    px0 = float(x0) / len(dataset)
    ig = entropy(target) - (px1 * calculate_entropy(px1c1, px1c0) +
                            px0 * calculate_entropy(px0c1, px0c0))
    return ig


def count_values(dataset, target, y):
    """
    Calculates the number of 1 and 0 and also the class where value is 1 or 0 of one column

    Args:
        target:
        dataset:
        y: value of the column

    Returns:
        x1: number of 1
        x0: number of 0
        x1c1: number of  class = 1 where x1
        x1c0: number of  class = 0 where x1
        x0c1: number of  class = 1 where x0
        x0c0: number of  class = 0 where x0

    """
    x1, x0, x1c1, x1c0, x0c1, x0c0 = 0, 0, 0, 0, 0, 0

    for txt in range(len(dataset)):
        if dataset[txt][y]:
            x1 += 1
            if target[txt]:
                x1c1 += 1
            else:
                x1c0 += 1
        else:
            x0 += 1
            if target[txt]:
                x0c1 += 1
            else:
                x0c0 += 1
    return x1, x0, x1c1, x1c0, x0c1, x0c0


def calculate_entropy(p0, p1):
    """
    Calculates the entropy of two possibilities
    Args:
        p0: possibility of 0
        p1: possibility of 1

    Returns: value of entropy

    """
    return - (p1 * math.log2(1 + p1) + p0 * math.log2(1 + p0))


