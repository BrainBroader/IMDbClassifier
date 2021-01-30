import numpy as np
import math
from RandomForest_clf.DecisionTree import DecisionTree
from analysis import vectorizing


class RandomForest:

    def __init__(self, n_trees=100, max_depth=math.inf, features=80, min_leaf=3):
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

            t = DecisionTree(dataset, target, self.max_depth, self.min_leaf)
            decision_tree = t.create_tree(dataset_index, vocabulary_index)
            self.trees.append(decision_tree)
            print("Tree number ", tree + 1, " of ", self.n_trees, "is fitted.")

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
        probabilities = np.zeros((len(documents), 2))
        for doc in range(vector.shape[0]):
            percents = np.mean([t.predict_class(vector[doc]) for t in self.trees], axis=0)
            predictions.append([1 if percents > 0.5 else 0])
            probabilities[doc, 0] = 1 - percents
            probabilities[doc, 1] = percents

        return predictions, probabilities
