import math

import numpy as np

from analysis import extract_vocabulary
from analysis import analyze


class MultinomialNaiveBayes:
    """ A representation of Multinomial Naive Bayes classifier.

    Attributes:
        vocabulary:
            A set of non-duplicate terms extracted from training data.
        targets:
            A set of target classes.
        prior:
            Prior probability of a document occurring in target class and is given by equation
                P(c) = Nc/N
            where Nc is the number of documents in class c and N is the total number of documents.
        cond_prob:
            Conditional probability of term tk occurring in a document of class c and is given by equation
                P(t|c) = (tf(t,c) + 1) / sum_t'(tf(c,t') + 1)
            where tf(t,c) is the term frequency of term t in training documents of class c.
    """

    def __init__(self):
        """ Initializes Multinomial Naive Bayes. """
        self.vocabulary = None
        self.targets = None
        self.prior = None
        self.cond_prob = None

    def fit(self, documents, targets):
        """ Trains Multinomial Naive Bayes according to training data and their target classes.

        Args:
            documents:
                A list of training documents as strings.
            targets:
                A list of target classes of the training data.
        """
        # extract vocabulary
        print(f'[DEBUG] - Extracting vocabulary...')
        self.vocabulary = list(extract_vocabulary(documents))
        # number of documents
        n_docs = len(documents)

        self.targets = set(targets)

        # priors of each target class
        # a vector of size: number of target classes
        self.prior = np.zeros((len(self.targets)))

        self.cond_prob = np.zeros((len(self.vocabulary), len(self.targets)))
        for target in self.targets:
            print(f'[DEBUG] - Counting documents in class {target}...')
            res = self.count_docs_in_class(documents, targets, target)
            n_docs_c = res[0]  # number of documents in target class
            docs_c = res[1]    # documents in class c

            # compute prior for target class
            print(f'[DEBUG] - Counting prior probability of class {target}...')
            self.prior[target] = n_docs_c/n_docs

            # term frequencies of vocabulary terms in documents of target class
            # a vector of size: number of vocabulary terms
            print(f'[DEBUG] - Calculating term frequencies of terms occurring in training documents of class {target}...')
            freq_t_c = self.term_frequency(docs_c)

            sum_freq = np.sum(freq_t_c) + freq_t_c.shape[0]

            # conditional probabilities of each term in target class
            # P(t|c) = (tf(t,c) + 1) / sum_t'(tf(c,t') + 1)
            # a matrix of size: number of vocabulary terms * number of target classes
            print(f'[DEBUG] - Calculating conditional probabilities of terms occurring in training documents of class {target}...')
            self.cond_prob[:, target] = (freq_t_c + 1)/sum_freq

    def predict(self, documents):
        """ Classifies the given documents.

        Args:
            documents:
                A list of documents as strings.

        Returns:
            A list with the target class of each document.
        """
        targets = []

        for document in documents:
            terms = extract_vocabulary([document])
            # keep terms that are in training vocabulary only
            terms.intersection_update(self.vocabulary)

            score = np.zeros((len(self.targets)))
            for target in self.targets:
                score[target] = math.log(self.prior[target], 2)

                for term in terms:
                    term_index = self.vocabulary.index(term)
                    score[target] = score[target] + math.log(self.cond_prob[term_index][target], 2)

            targets.append(np.argmax(score))

        return targets

    def count_docs_in_class(self, documents, targets, target):
        """ Counts documents in specified target class.

        Args:
            documents:
                A list of documents as strings.
            targets:
                The target classes of documents.
            target:
                The target class.

        Returns:
            A tuple with the number of documents and a list of this documents.
        """
        docs = []
        counter = 0

        for index in range(len(documents)):
            if targets[index] == target:
                docs.append(documents[index])
                counter += 1

        return counter, docs

    def term_frequency(self, documents):
        """ Finds the term frequencies of the vocabulary terms occurring in the given set of documents.

        Args:
            documents:
                A list of documents as strings.

        Returns:
            A np.array with the term frequencies.
            The array if a vector with size: the number of vocabulary terms
        """
        freq_t_c = np.zeros(len(self.vocabulary))

        for document in documents:
            tokens = analyze(document)
            for token in tokens:
                if token in self.vocabulary:
                    term_index = self.vocabulary.index(token)
                    freq_t_c[term_index] += 1
        return freq_t_c
