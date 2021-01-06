from nltk.corpus import stopwords


def extract_vocabulary(data):
    vocab = set()

    for document in data:
        tokens = set(document.split())
        vocab.update(tokens)
    print(len(vocab))
    stop_words = set(stopwords.words('english'))
    vocab.difference_update(stop_words)
    print(len(vocab))


    return vocab


