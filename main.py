import os
import sys

import sklearn.model_selection
from dochandler import extract_vocabulary, create_vector, document_pre_processing


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


train_path = sys.argv[1] + '\\train\\'
test_data_path = sys.argv[1] + '\\test\\'

# load training data
print(f'[INFO] - Loading training data from {train_path}')
res = read_data(train_path)
train_data = res[0]
train_target = res[1]
print(f'[INFO] - Total training data files {len(train_data)} and target classes {len(train_target)}')

# 10% of training data will go to developer data set
print(f'[INFO] - Splitting training data into training data and developer data (10% of training data)')
res = split_data(train_data, train_target, 0.1)
train_data = res[0]
train_target = res[2]
print(f'[INFO] - Total training data files {len(train_data)} and target classes {len(train_target)}')
dev_data = res[1]
dev_target = res[3]
print(f'[INFO] - Total developer data files {len(dev_data)} and target classes {len(dev_target)}')



