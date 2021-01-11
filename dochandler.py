from nltk.corpus import stopwords


# TODO:remove # before sending assignment
import nltk

# nltk.download(all)


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
        tokens = document_pre_processing(document)
        vocab.update(tokens)

    return vocab


def create_vector(document, vocab):
    """

    Args:
        document:
        vocab:

    Returns:

    """
    vector = []
    tokens = document_pre_processing(document)
    for word in vocab:
        if word in tokens:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def document_pre_processing(document):
    """

    Args:
        document:

    Returns:

    """

    # document.translate(str.maketrans(", ", string.punctuation))
    document = document.lower()

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = set(tokenizer.tokenize(document))

    stop_words = set(stopwords.words('english'))
    tokens.difference_update(stop_words)

    return tokens
