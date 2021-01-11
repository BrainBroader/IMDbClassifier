import numpy as np

from dochandler import extract_vocabulary


class MultinomialNaiveBayes:

    def fit(self, documents, targets):
        vocabulary = extract_vocabulary(documents[:, 0])
        n_docs = documents.shape[0]  # number od documents

        prior = {}  # priors of each class # FIXME return as np.array
        # c1 - prior_c1
        # c2 - prior_c2

        cond_prob = np.array() # FIXME find sizes && dimensions are the same for each class since we find prob for t in both classes
        # t_ct = {}

        for target in targets:
            res = self.count_docs_in_class(documents, target)

            n_docs_c = res[0]  # number of documents in class c
            docs_c = res[1]  # documents in class c

            prior[target] = n_docs_c/n_docs

            # FIXME see if can be done better
            freq_t_c = self.term_frequency(docs_c, vocabulary)  # term frequencies of class c  # FIXME np.array (do it in method)

            # FIXME see if can be done better
            summary = 0
            for temp in freq_t_c:
                summary += freq_t_c[temp] + 1

            for term in vocabulary:
                cond_prob[(t, target)] = (t_ct[(t, target)] + 1)/summary  # (?) quick array division w/ np

        return vocabulary, prior, cond_prob

    def count_docs_in_class(self, documents, target):
        """ Counts documents in specified target class.

        Args:
            documents:
                A numpy array with documents in 1st column and their target classes in 2nd column.
            target:
                The target class

        Returns:
            A tuple with the number of documents and a list of this documents.
        """
        docs = []
        counter = 0

        for doc in documents:
            if doc[1].astype('int') == target:  # column with numbers is str cause of first column
                docs.append(doc[0])
                counter += 1

        return counter, docs

    def term_frequency(self, documents, vocabulary):
        freq_t_c = {}
        for document in documents:
            tokens = document.split()
            for token in tokens:
                if token in vocabulary:
                    if token in freq_t_c:
                        freq_t_c[token] = freq_t_c.get(token) + 1
                    else:
                        freq_t_c[token] = 1
        return freq_t_c
