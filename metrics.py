from sklearn.metrics import precision_recall_curve

import numpy as np


def accuracy(targets_true, targets_predict):
    counter = 0
    for index in range(len(targets_true)):
        if targets_true[index] == targets_predict[index]:
            counter += 1

    return counter / len(targets_true)


def precision_recall(probabilities, targets_true):
    precision = dict()
    recall = dict()
    thresholds = []

    average_precision = []
    average_recall = []

    for i in range(len(set(targets_true))):
        precision[i], recall[i], thresholds = precision_recall_curve(targets_true, probabilities[:, i])


    precisions_0 = precision[0]
    precisions_1 = precision[1]

    recall_0 = recall[0]
    recall_1 = recall[1]

    for index in range(len(thresholds)):
        average_precision.append((precisions_0[index] + precisions_1[index]) / 2)
        average_recall.append((recall_0[index] + recall_1[index]) / 2)

    return average_precision, average_recall, thresholds


def f1(precision, recall):
    f1_score = np.zeros(len(precision))

    for index in range(len(precision)):
        f1_score[index] = (2 * precision[index] * recall[index]) / (precision[index] + recall[index])

    return f1_score
