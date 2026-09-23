import math
from collections import Counter
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    # Write code here
    n = len(documents)
    doc_splits = [doc.split() for doc in documents]
    vocabulary = sorted(list({word for doc in doc_splits for word in doc}))
    
    df = Counter(word for doc in doc_splits for word in set(doc))

    tfidf = [
        [(doc.count(word) / len(doc)) * math.log(n / df[word]) for word in vocabulary]
        for doc in doc_splits
    ]

    # tf = []
    # for doc in documents:
    #     tmp = []
    #     d = len(doc.split())
    #     for word in vocabulary:
    #         tmp.append(doc.count(word) / d)
    #     tf.append(tmp)
          
    # idf = {t: math.log(n / d) for t,d in df.items()}
    # tfidf = tf
    # for i, doc in enumerate(documents):
    #     for j, word in enumerate(vocabulary):
    #         tfidf[i][j] *= idf[word]

    return {
        "tfidf_matrix": np.asarray(tfidf),
        "vocabulary": vocabulary
    }