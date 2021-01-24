import os
import sys

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from naive_bayes_clf import MultinomialNaiveBayes

from metrics import accuracy, precision_recall, f1


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

    nb = MultinomialNaiveBayes()

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

        feature_size = 0.007
        print(f'[INFO] - Fitting Multinomial Naive Bayes classifier using {feature_size*100:.1f}% of features...')
        nb.fit(fold_data, fold_target, feature_size)

        print(f'[INFO] - Predicting with Multinomial Naive Bayes classifier using train data...')
        nb_targets, _ = nb.predict(fold_data)
        accuracy_score = accuracy(fold_target, nb_targets)
        accuracy_train.append(accuracy_score)
        print(f'[INFO] - Accuracy: {accuracy_score}')

        print(f'[INFO] - Predicting with Multinomial Naive Bayes classifier using developer data...')
        nb_targets, _ = nb.predict(dev_data)
        accuracy_score = accuracy(dev_target, nb_targets)
        print(f'[INFO] - Accuracy: {accuracy_score}')

        print(f'[INFO] - Predicting with Multinomial Naive Bayes classifier using test data...')
        nb_targets, probabilities = nb.predict(test_data)
        accuracy_score = accuracy(test_target, nb_targets)
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
    average_precision, average_recall, thresholds = precision_recall(probabilities, test_target, 100)
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
