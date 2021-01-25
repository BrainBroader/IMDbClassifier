import numpy as np
import math
from RandomForest.DecisionTree import Tree
from dochandler import vectorizing


class RandomForest:

    def __init__(self, n_trees, features=100, max_depth=math.inf, min_leaf=3):
        # np.random.seed(42)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.features = features
        self.trees = []

    def fit(self, data, target, vocabulary):
        """
        Trains the model.
        Chooses in random and with replace rows of a starting dataset and creates different datasets with the same size
        as the starting dataset.
        Then chooses in random a number of features from the starting list of featurees.
        Then creates multiple trees using these datasets and vocabularies.

        """

        dataset = np.array(vectorizing(data, vocabulary))
        target = np.array(target)

        for tree in range(self.n_trees):

            dataset_index = np.random.choice(len(dataset), replace=True, size=len(dataset))
            vocabulary_index = np.random.choice(len(vocabulary), size=self.features)

            t = Tree(dataset, target, self.max_depth, self.min_leaf)
            tree = t.create_tree(dataset_index, vocabulary_index)
            self.trees.append(tree)
            print("Tree number ", tree, " fitted.")

    def predict(self, documents, vocabulary):
        """
        Gets the class prediction of each tree.

        Args:
            vocabulary: the starting vocabulary.
            documents: a group of given documents.

        Returns:
            A list with classes that had more "votes" for each document.

        """
        vector = np.array(vectorizing(documents, vocabulary))
        predictions = []
        for doc in vector:
            percents = np.mean([t.predict_class(doc) for t in self.trees], axis=0)
            predictions.append([1 if percents > 0.5 else 0])
        return predictions
