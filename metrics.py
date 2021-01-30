from math import floor

from sklearn.metrics import precision_recall_curve

import numpy as np


def accuracy(targets_true, targets_predict):
    """ Computes accuracy classification score.
    Accuracy is the fraction of correctly classified documents.
    Args:
        targets_true:
            The correct target classes, as a list.
        targets_predict:
            The predicted target classes by a classifier, as a list.
    Returns:
        The accuracy classification score, as a float.
    """
    counter = 0
    for index in range(len(targets_true)):
        if targets_true[index] == targets_predict[index]:
            counter += 1

    return counter / len(targets_true)


def precision_recall(probabilities, targets_true, n_thresholds):
    """ Computes precision-recall curve using a given number thresholds.
    Args:
        probabilities:
            Predicted probabilities, as an array of size (number of documents, number of target classes)
        targets_true:
            The correct target classes, as a list.
        n_thresholds:
            Number of thresholds to be used, as integer.
    Returns:
        precision array of length (n_thresholds),
        recall array of length (n_thresholds),
        thresholds as an array
    """
    precision = dict()
    recall = dict()
    temp_thresholds = []

    for i in range(len(set(targets_true))):
        precision[i], recall[i], temp_thresholds = precision_recall_curve(targets_true, probabilities[:, i])

    temp_pre_0 = precision[0]
    temp_pre_1 = precision[1]
    temp_rec_0 = recall[0]
    temp_rec_1 = recall[1]

    # keep only the given number thresholds and their precision-recall
    precision_0 = []
    precision_1 = []
    recall_0 = []
    recall_1 = []
    thresholds = []
    step = floor(temp_thresholds.shape[0] / n_thresholds)
    index = 0
    for i in range(n_thresholds):
        thresholds.append(temp_thresholds[index])
        precision_0.append(temp_pre_0[index])
        precision_1.append(temp_pre_1[index])
        recall_0.append(temp_rec_0[index])
        recall_1.append(temp_rec_1[index])
        index = index + step

    average_precision = []
    average_recall = []

    for index in range(len(thresholds)):
        average_precision.append((precision_0[index] + precision_1[index]) / 2)
        average_recall.append((recall_0[index] + recall_1[index]) / 2)

    return average_precision, average_recall, thresholds


def f1(precision, recall):
    """ Computes f1 measure.
    Args:
        precision:
            Precision values, as a list.
        recall:
            Recall values, as a list.
    Returns:
        F1 values for each precision-recall values.
    """
    f1_score = np.zeros(len(precision))

    for index in range(len(precision)):
        f1_score[index] = (2 * precision[index] * recall[index]) / (precision[index] + recall[index])

    return f1_score
