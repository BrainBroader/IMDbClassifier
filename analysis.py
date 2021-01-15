from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

# TODO: remove comments before sending assignment
# import nltk
# nltk.download('stopwords')


def extract_vocabulary(documents):
    """ Extracts vocabulary from given set of documents.

    Args:
        documents:
            A list of documents as strings.

    Returns:
        The vocabulary set.
    """
    vocabulary = set()

    for document in documents:
        tokens = analyze(document)
        vocabulary.update(tokens)

    return vocabulary


def analyze(document):
    """ Analyzes a document.

    The document is transformed to lower case and tokenized.
    Then punctuation and stopwords are removed. Also, stemming is used.

    Args:
        document:
            A document as a string.

    Returns:
        A set of extracted tokens from the document.
    """
    # document to lower case
    document = document.lower()

    # tokenize document and keep only alphanumeric characters
    tokenizer = RegexpTokenizer('[a-zA-Z0-9]+')
    tokens = set(tokenizer.tokenize(document))

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens.difference_update(stop_words)

    # stemming words
    stemmer = PorterStemmer()

    stem_tokens = set()
    for token in tokens:
        stem_tokens.add(stemmer.stem(token))

    return stem_tokens


def create_vector(document, vocabulary):
    """ Creates a vector of attributes for a document.

    The vector is of vocabulary size and each position is 1 if the document term is in vocabulary, otherwise 0.

    Args:
        document:
            A document as a string.
        vocabulary:
            A set of terms from documents of training collection.

    Returns:
        The document vector.
    """
    vector = []
    tokens = analyze(document)

    for word in vocabulary:
        if word in tokens:
            vector.append(1)
        else:
            vector.append(0)
    return vector
