import os
import sys

import sklearn.model_selection
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from metrics import precision_recall, f1, accuracy
from RandomForest_clf.RandomForest import RandomForest
from analysis import frequent_features


def read_data(path):

    """ Reads the data files from aclImdb folder, and keeps the target class, negative or positive, of each data file.

    Args:
        path:
            The path to aclImdb folder as a string.

    Returns:
        A tuple with the list of the data as strings and a list with the target classes as integers.
    """
    data = []
    target = []
    for folder in os.listdir(path):
        if (os.path.isdir(path + folder)) and (folder != 'unsup'):
            for filename in os.listdir(path + folder):
                file = open(path + folder + '\\' + filename, 'r', encoding='utf-8')
                data.append(file.read())
                if folder == 'pos':
                    # denote positive class with 1
                    target.append(1)
                elif folder == 'neg':
                    # denote negative class with 0
                    target.append(0)
                file.close()
    return data, target


def split_data(data, target, size):
    """ Splits training set in two pieces; a training set and a developer set.

    The training set should constitute of the data files and the target classes of each data file.

    Args:
        data:
            A list with the contents of each data file.
        target:
            A list of the target classes of each data file.
        size:
            The size of the developer set as a float (ex 0.1 means 10% of the training set)

    Returns:
        The training data, the developer data, the target classes for training data and the target classes for developer
        data.
    """
    return sklearn.model_selection.train_test_split(data, target, test_size=size)


def main():
    train_path = sys.argv[1] + '\\train\\'
    test_path = sys.argv[1] + '\\test\\'

    # load training data
    print(f'[INFO] - Loading training data from {train_path}')
    res = read_data(train_path)
    train_data = res[0]
    train_target = res[1]
    print(f'[INFO] - Total train data: {len(train_data)}')

    print(f'[INFO] - Loading testing data from {test_path}')
    res = read_data(test_path)
    test_data = res[0]
    test_target = res[1]
    print(f'[INFO] - Total test data: {len(test_data)}')

    # 10% of training data will go to developer data set
    print(f'[INFO] - Splitting training data into training data and developer data (keeping 10% for training data)')
    res = train_test_split(train_data, train_target, test_size=0.1)
    train_data = res[0]
    train_target = res[2]
    print(f'[INFO] - Total training data after split {len(train_data)}')
    dev_data = res[1]
    dev_target = res[3]
    print(f'[INFO] - Total developer data {len(dev_data)}')

    rf = RandomForest(100, 10)

    accuracy_train = []
    accuracy_test = []

    counter = 1
    for train_size in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f'\n[INFO] - Iteration No.{counter} (using {int(train_size*100)}% of 90% of train data).')

        if train_size != 1.0:
            res = train_test_split(train_data, train_target, train_size=train_size, shuffle=False)
            fold_data = res[0]
            fold_target = res[2]
        else:
            fold_data = train_data
            fold_target = train_target

        feature_size = 100
        vocabulary = frequent_features(train_data, feature_size)
        print(f'[INFO] - Fitting Random forest classifier using', feature_size, ' features...')
        rf.fit(fold_data, fold_target, vocabulary)

        print(f'[INFO] - Predicting with Random Forest classifier using train data...')
        rf_targets, _ = rf.predict(fold_data, vocabulary)
        accuracy_score = accuracy(fold_target, rf_targets)
        accuracy_train.append(accuracy_score)
        print(f'[INFO] - Accuracy: {accuracy_score}')

        print(f'[INFO] - Predicting with Random Forest classifier using developer data...')
        rf_targets, _ = rf.predict(dev_data, vocabulary)
        accuracy_score = accuracy(dev_target, rf_targets)
        print(f'[INFO] - Accuracy: {accuracy_score}')

        print(f'[INFO] - Predicting with Random Forest classifier using test data...')
        rf_targets, probabilities = rf.predict(test_data, vocabulary)
        accuracy_score = accuracy(test_target, rf_targets)
        accuracy_test.append(accuracy_score)
        print(f'[INFO] - Accuracy: {accuracy_score}')

        counter += 1

    learning_curves_plot = plt.figure(1)
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], accuracy_train, label='train')
    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], accuracy_test, label='test')
    plt.title('Learning Curves (Multinomial Naive Bayes)')
    plt.legend(loc='lower right')
    plt.xlabel('Number of Train Data')
    plt.ylabel('Accuracy')

    precision_recall_plot = plt.figure(2)
    average_precision, average_recall, thresholds = precision_recall(probabilities, test_target, 10)
    plt.step(average_recall, average_precision, where='post')
    plt.title('Precision-Recall Curve (Multinomial Naive Bayes)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    f1_plot = plt.figure(3)
    f1_score = f1(average_precision, average_recall)
    plt.plot(thresholds, f1_score)
    plt.title('F1 Curve (Multinomial Naive Bayes)')
    plt.xlabel('Thresholds')
    plt.ylabel('F1 Measure')

    plt.show()


if __name__ == '__main__':
    main()
