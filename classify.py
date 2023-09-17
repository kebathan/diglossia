"""
Train a Gaussian Naive Bayes classifier on distinguishing literary and colloquial Tamil.

Author: Aryaman Arora
Date: 2023-09-10
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from collections import defaultdict
import numpy as np

def featurise(sents, char_n_max: int = 3, word_n_max: int = 3):
    """Featurise a list of sentences into a list of lists of n-grams."""

    # clean sents (remove punctuation, etc.)
    sents = [''.join([x for x in sent.lower().replace(' ', '_') if x not in '.,']) for sent in sents]

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
            for i in range(len(sent.split('_')) - n + 1):
                word_ngrams[-1].append(label_to_id["w#" + "_".join(sent.split('_')[i:i+n])])
    
    # convert n-grams to counts
    features = []
    for i in range(len(sents)):
        features.append([0 for _ in range(len(label_to_id))])
        for ngram in char_ngrams[i]:
            features[-1][ngram] += 1
        for ngram in word_ngrams[i]:
            features[-1][ngram] += 1
    
    # make id to label
    id_to_label = {}
    for label in label_to_id:
        id_to_label[label_to_id[label]] = label
    
    return features, label_to_id, id_to_label


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
    X, label_to_id, id_to_label = featurise(X_raw, char_n_max=0, word_n_max=1)

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
    print("\nFirst couple predictions:")
    for i in range(3):
        print(X_raw[i])
        print("Predicted: " + y_pred[i])
        print("Actual: " + y_test[i])
        print()
    
    # print most informative features
    # Find the features with the largest differences in means between classes
    mean_diffs = gnb.theta_[0, :] - gnb.theta_[1, :]
    abs_mean_diffs = np.abs(mean_diffs)
    sorted_mean_diffs = np.argsort(abs_mean_diffs)[::-1]

    print("Most informative features (positive = colloquial):")
    for i in range(30):
        print(f"{id_to_label[sorted_mean_diffs[i]]:<20} {mean_diffs[sorted_mean_diffs[i]]:>8.4f}")

def main():
    train_model()

if __name__ == "__main__":
    main()