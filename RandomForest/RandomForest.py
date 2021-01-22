import numpy as np
from RandomForest.DecisionTree import Tree


class RandomForest:

    def __init__(self, n_trees, max_depth=5, min_leaf=3):

        # np.random.seed(42)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.trees = []

    def fit(self, dataset, target):

        """
        Trains the model.
        Chooses in random and with replace rows of a starting dataset and creates different datasets with the same size
        as the starting dataset.
        Then creates multiple trees using these datasets.

        """
        for tree in range(self.n_trees):
            shuffled_dataset = []
            c = []
            data = np.random.choice(len(dataset), replace=True, size=len(dataset))
            for doc in data:
                shuffled_dataset.append(dataset[doc])
                c.append(target[doc])
            t = Tree(self.max_depth, self.min_leaf)
            tree = t.create_tree(shuffled_dataset, c)
            self.trees.append(tree)
            print("///////////////////////////////")

    def predict(self, document):

        """
        Gets the class prediction of each tree.

        Args:
            document: a given document.

        Returns:
            The class that had more "votes" for this document.

        """
        percents = np.mean([t.predict_class(document) for t in self.trees], axis=0)
        return [1 if percents > 0.5 else 0]


