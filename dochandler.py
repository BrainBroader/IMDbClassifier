from nltk.corpus import stopwords

# TODO:remove # before sending assignment
# import nltk
# nltk.download('stopwords')


def extract_vocabulary(documents):
    """ Extracts vocabulary from given set of documents, removes duplicates and stopwords

    Args:
        documents:
            A list of documents as strings

    Returns:
        The vocabulary set
    """
    vocab = set()

    for document in documents:
        tokens = set(document.split())
        vocab.update(tokens)

    # stop_words = set(stopwords.words('english'))
    # vocab.difference_update(stop_words)

    return vocab
