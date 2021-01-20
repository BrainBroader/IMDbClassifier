import heapq as pq
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
                P(t|c) = (df(t,c) + 1) / sum_t'(df(t',c) + 1)
            where:
                df(t,c) is the document frequency of term t in training documents of class c.
        tokenized_docs:
            A vector of lists with tokens of each training document.
        n_docs:
            Number of training documents.
        n_docs_c:
            A vector with the number of documents in each target class.
        docs_c:
            A vector of lists with the training documents (corresponding to tokenized_docs) in each target class.
    """

    def __init__(self):
        """ Initializes Multinomial Naive Bayes. """
        self.vocabulary = None
        self.targets = None
        self.prior = None
        self.cond_prob = None

        self.tokenized_docs = None

        self.n_docs = 0
        self.n_docs_c = None
        self.docs_c = None

    def fit(self, documents, targets, k):
        """ Trains Multinomial Naive Bayes according to training data and their target classes.

        Args:
            documents:
                A list of training documents as strings.
            targets:
                A list of target classes of the training data.
            k:
                Number of features to be used, as a float in range (0, 1] (ex. 0.1 means 10% of features)
        """
        self.targets = set(targets)
        self.n_docs = len(documents)
        self.n_docs_c = [0] * len(self.targets)
        self.docs_c = [[]] * len(self.targets)
        for target in self.targets:
            res = self.count_docs_in_class(documents, targets, target)
            self.n_docs_c[target] = res[0]
            self.docs_c[target] = res[1]

        self.vocabulary = list(extract_vocabulary(documents))
        self.vocabulary = self.feature_selection(documents, targets, self.vocabulary, k)

        # priors of each target class
        # a vector of size: number of target classes
        self.prior = np.zeros((len(self.targets)))
        self.cond_prob = np.zeros((len(self.vocabulary), len(self.targets)))
        for target in self.targets:
            # compute prior for target class
            self.prior[target] = self.n_docs_c[target]/self.n_docs

            # df of vocabulary terms of documents in target class
            # a vector of size: number of vocabulary terms
            df_t_c = self.df(self.docs_c[target])

            sum_freq = np.sum(df_t_c) + df_t_c.shape[0]

            # conditional probabilities of each term in target class
            # P(t|c) = (idf(t,c) + 1) / sum_t'(idf(t',c) + 1)
            # a matrix of size: number of vocabulary terms * number of target classes
            self.cond_prob[:, target] = (df_t_c + 1)/sum_freq

    def predict(self, documents):
        """ Classifies the given documents.

        Args:
            documents:
                A list of documents as strings.

        Returns:
            A list with the target class of each document.
        """
        targets = []
        probabilities = np.zeros((len(documents), len(self.targets)))

        for index_doc in range(len(documents)):
            terms = extract_vocabulary([documents[index_doc]])
            # keep terms that are in training vocabulary only
            terms.intersection_update(self.vocabulary)

            for target in self.targets:
                score = math.log2(self.prior[target])

                for term in terms:
                    term_index = self.vocabulary.index(term)
                    score = score + math.log2(self.cond_prob[term_index][target])
                    probabilities[index_doc, target] = score

            targets.append(np.argmax(probabilities[index_doc][:]))

        return targets, probabilities

    def feature_selection(self, documents, targets, vocabulary, k):
        """ Selects top k featured from vocabulary using mutual information.

        Args:
            documents:
                A list of training documents as strings.
            targets:
                A list of target classes of the training data.
            vocabulary:
                The vocabulary from the training documents
            k:
                Number of features to be returned, as float in range (0, 1] (ex. 0.1 means 10% of features)

        Returns:
            A list with the top k features.
        """
        features = dict()
        k = int(math.floor(len(vocabulary) * k))

        self.tokenized_docs = [None] * self.n_docs

        class_entropy = 0
        for target in self.targets:
            class_entropy -= (self.n_docs_c[target]/self.n_docs) * math.log2(self.n_docs_c[target]/self.n_docs)

        for term in vocabulary:
            utility = self.mutual_information(documents, targets, term)
            features[term] = class_entropy - utility

        return pq.nlargest(k, features, key=features.get)

    def mutual_information(self, documents, targets, term):
        """ Computes mutual information for a given term.

        Args:
            documents:
                A list of training documents as strings.
            targets:
                A list of target classes of the training data.
            term:
                A term as a string.
        Returns:
            The mutual information of the given term, as a float.
        """
        # each row represents the times a term occurs (or doesn't occur)
        # each column represents the class of the document where the term occurs (or doesn't occur)
        n_docs_t_c = np.zeros((2, len(set(targets))))
        for index in range(self.n_docs):
            # if document hasn't been tokenized yet, tokenize it
            # this is used to minimize the time taken by this function
            if self.tokenized_docs[index] is None:
                self.tokenized_docs[index] = analyze(documents[index])

            tokens = self.tokenized_docs[index]
            if term in tokens:
                if targets[index] == 0:
                    n_docs_t_c[1][0] += 1
                else:
                    n_docs_t_c[1][1] += 1
            else:
                if targets[index] == 0:
                    n_docs_t_c[0][0] += 1
                else:
                    n_docs_t_c[0][1] += 1

        mi = 0
        for row in range(n_docs_t_c.shape[0]):
            for col in range(n_docs_t_c.shape[1]):
                try:
                    mi += (n_docs_t_c[row][col] / self.n_docs) * math.log2((self.n_docs * n_docs_t_c[row][col]) / (self.n_docs_c[row] * self.n_docs_c[col]))
                except ValueError:
                    # when x in log(x) is zero, add 0 to mutual information
                    mi += 0

        return mi

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
            A tuple with the number of documents in target class and a list of indexes of these documents.
        """
        docs = []
        counter = 0

        for index in range(len(documents)):
            if targets[index] == target:
                docs.append(index)
                counter += 1

        return counter, docs

    def df(self, documents):
        """ Finds inverse document frequencies of the vocabulary terms occurring in the given set of documents.

        Args:
            documents:
                A list of documents as strings.

        Returns:
            A np.array with the document frequencies.
            The array as a vector with size: the number of vocabulary terms
        """
        df_t_c = np.zeros(len(self.vocabulary))

        for index in documents:
            tokens = self.tokenized_docs[index]
            for token in tokens:
                if token in self.vocabulary:
                    term_index = self.vocabulary.index(token)
                    df_t_c[term_index] += 1

        return df_t_c