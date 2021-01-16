import numpy as np
from DecisionTree import DecisionTree


class RandomForest:

    def __init__(self, dataset, vocabulary, target, n_trees, max_depth=5):

        np.random.seed(42)
        self.dataset = dataset
        self.vocabulary = list(vocabulary)
        self.target = target
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self):

        """
        Trains the model.
        Chooses in random and with replace rows of a starting dataset and creates different datasets with the same size
        as the starting dataset.
        Then creates multiple trees using these datasets.

        """
        for tree in range(self.n_trees):
            dataset = []
            c = []
            data = np.random.choice(len(self.dataset), replace=True, size=len(self.dataset))
            for doc in data:
                dataset.append(self.dataset[doc])
                c.append(self.target[doc])
            t = DecisionTree(dataset, self.vocabulary, c, self.max_depth)
            t.create_tree()
            self.trees.append(t)

    def predict(self, document):

        """
        Gets the class prediction of each tree.

        Args:
            document: a given document.

        Returns:
            The class that had more "votes" for this document.

        """
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict_class(document))
        if np.count_nonzero(predictions) > self.n_trees/2:
            return 1
        else:
            return 0
