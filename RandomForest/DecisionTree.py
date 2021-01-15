import numpy as np


class DecisionTree:

    def __init__(self, data, vocab, c, max_depth, depth=0):
        self.data = data
        self.vocab = list(vocab)
        self.c = c
        self.max_depth = max_depth
        self.depth = depth
        self.left = None
        self.right = None
        self.best_split = None

    def create_tree(self):
        if len(self.data) == 0:
            return 1
        elif all(x == self.c[0] for x in self.c):
            return self.c[0]
        elif len(self.vocab) == 0:
            if np.count_nonzero(self.c) >= len(self.c) / 2:
                return 1
            else:
                return 0
        elif self.depth <= self.max_depth:
            self.depth += 1
            self.best_split = self.choose_attribute()
            data_left, c_left, data_right, c_right = self.create_table(self.best_split)

            self.vocab.pop(self.best_split)

            self.left = DecisionTree(data_left, self.vocab, c_left, self.max_depth, self.depth)
            self.right = DecisionTree(data_right, self.vocab, c_right, self.max_depth, self.depth)

            self.left.create_tree()
            self.right.create_tree()

    def create_table(self, best_split):

        data_left = []
        c_left = []
        data_right = []
        c_right = []
        for txt in range(len(self.data)):
            if self.data[txt][best_split]:
                data_left.append(self.data[txt])
                c_left.append(self.c[txt])
            else:
                data_right.append(self.data[txt])
                c_right.append(self.c[txt])
        data_left = np.delete(data_left, best_split, axis=1)
        data_right = np.delete(data_right, best_split, axis=1)
        return data_left, c_left, data_right, c_right

    def choose_attribute(self):
        maximum = 0
        pos = 0

        for attr in range(len(self.vocab)):
            ig = self.information_gain(attr)
            if ig > maximum:
                maximum = ig
                pos = attr
        return pos

    def entropy(self):
        c1 = np.count_nonzero(self.c)
        c0 = len(self.c) - c1
        return self.calculate_entropy(c1 / len(self.c), c0 / len(self.c))

    def information_gain(self, y):
        x1, x0, x1c1, x1c0, x0c1, x0c0 = self.count_values(y)
        if x1 and x0:
            px1c1 = x1c1 / x1
            px1c0 = x1c0 / x1
            px0c1 = x0c1 / x0
            px0c0 = x0c0 / x0
            px1 = x1 / len(self.data)
            px0 = x0 / len(self.data)
            ig = self.entropy() - (px1 * self.calculate_entropy(px1c1, px1c0) +
                                   px0 * self.calculate_entropy(px0c1, px0c0))
            return ig
        else:
            return 0

    def count_values(self, y):
        x1, x0, x1c1, x1c0, x0c1, x0c0 = 0, 0, 0, 0, 0, 0

        for txt in range(len(self.data)):
            if self.data[txt][y]:
                x1 += 1
                if self.c[txt]:
                    x1c1 += 1
                else:
                    x1c0 += 1
            else:
                x0 += 1
                if self.c[txt]:
                    x0c1 += 1
                else:
                    x0c0 += 1
        return x1, x0, x1c1, x1c0, x0c1, x0c0

    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print(self.best_split)
        if self.right:
            self.right.print_tree()

    @staticmethod
    def calculate_entropy(p0, p1):
        return - (p1 * np.log2(1 + p1) + p0 * np.log2(1 + p0))
