from dochandler import extract_vocabulary


class MultinomialNaiveBayes:

    def fit(self, documents, targets):
        vocabulary = extract_vocabulary(documents)
        number_of_docs = len(documents)

        prior = {}
        cond_prob = {}
        t_ct = {}

        for target in set(targets):
            n_c = self.count_docs_in_class(targets, target)
            prior[target] = n_c/number_of_docs
            text_c = self.concatenate_text_of_all_docs_in_class(documents, targets, target)
            for t in vocabulary:
                t_ct[(t, target)] = self.count_tokens_of_term(text_c, t)

            summary = 0
            for temp in t_ct.keys():
                if temp[1] == target:
                    summary += t_ct[temp] + 1

            for t in vocabulary:
                cond_prob[(t, target)] = (t_ct[(t, target)] + 1)/summary

        return vocabulary, prior, cond_prob

    def count_docs_in_class(self, targets, target):
        counter = 0
        for document in targets:
            if document == target:
                counter += 1
        return counter

    def concatenate_text_of_all_docs_in_class(self, documents, targets, target):
        string = ''
        for document in range(len(documents)):
            if targets[document] == target:
                string += documents[document]
        return string

    def count_tokens_of_term(self, text, t):
        counter = 0
        for w in text.split():
            if w == t:
                counter += 1
        return counter