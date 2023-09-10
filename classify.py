"""
Train a Gaussian Naive Bayes classifier on distinguishing literary and colloquial Tamil.

Author: Aryaman Arora
Date: 2023-09-10
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from collections import defaultdict

def featurise(sents: list[str], char_n_max: int = 3, word_n_max: int = 3):
    """Featurise a list of sentences into a list of lists of n-grams."""

    # label to id and id to label
    label_to_id = defaultdict(lambda: len(label_to_id))

    # make char n-grams
    char_ngrams = []
    for sent in sents:
        char_ngrams.append([])
        for n in range(1, char_n_max + 1):
            for i in range(len(sent) - n + 1):
                char_ngrams[-1].append(label_to_id[sent[i:i+n]])
    
    # make word n-grams
    word_ngrams = []
    for sent in sents:
        word_ngrams.append([])
        for n in range(1, word_n_max + 1):
            for i in range(len(sent.split()) - n + 1):
                word_ngrams[-1].append(label_to_id["w_" + " ".join(sent.split()[i:i+n])])
    
    # convert n-grams to counts
    features = []
    for i in range(len(sents)):
        features.append([0 for _ in range(len(label_to_id))])
        for ngram in char_ngrams[i]:
            features[-1][ngram] += 1
        for ngram in word_ngrams[i]:
            features[-1][ngram] += 1
    
    return features


def train_model():
    """Train a Gaussian Naive Bayes classifier on the data."""

    # read data
    data = pd.read_csv("data/regdataset.csv")

    # make X (sentences) and y (labels)
    literary = data["transliterated"].tolist()
    colloquial = data["colloquial: annotator 1"].tolist() + data["colloquial: annotator 2"].tolist()
    X_raw = literary + colloquial
    y = (["literary"] * len(literary)) + (["colloquial"] * (len(colloquial)))

    # featurise
    X = featurise(X_raw, char_n_max=3, word_n_max=1)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create a Gaussian Naive Bayes classifier + train
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # predict
    y_pred = gnb.predict(X_test)

    # print results
    print("Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))

    # print first couple predictions
    for i in range(10):
        print(X_raw[i])
        print("Predicted: " + y_pred[i])
        print("Actual: " + y_test[i])
        print()

def main():
    train_model()

if __name__ == "__main__":
    main()