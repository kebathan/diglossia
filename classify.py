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
import pickle

def featurise(
    sents: list[str],
    char_n_max: int = 3,
    word_n_max: int = 3,
    label_to_id: dict[str, int] = None,
    id_to_label: dict[int, str] = None
):
    """Featurise a list of sentences into a list of lists of n-grams."""

    # clean sents (remove punctuation, etc.)
    sents = [''.join([x for x in sent.lower().replace(' ', '_') if x not in '.,']) for sent in sents]

    # label to id and id to label
    provided_labels = True
    if label_to_id is None:
        provided_labels = False
        label_to_id = defaultdict(lambda: len(label_to_id))

    # make char n-grams
    char_ngrams = []
    for sent in sents:
        char_ngrams.append([])
        for n in range(1, char_n_max + 1):
            for i in range(len(sent) - n + 1):
                key = sent[i:i+n]
                if provided_labels and key not in label_to_id:
                    continue
                char_ngrams[-1].append(label_to_id[key])
    
    # make word n-grams
    word_ngrams = []
    for sent in sents:
        sent_split = list(sent.split('_'))
        word_ngrams.append([])
        for n in range(1, word_n_max + 1):
            for i in range(len(sent_split) - n + 1):
                key = "w#" + "_".join(sent_split[i:i+n])
                if provided_labels and key not in label_to_id:
                    continue
                word_ngrams[-1].append(label_to_id[key])
    
    # convert n-grams to counts
    features = []
    for i in range(len(sents)):
        features.append([0 for _ in range(len(label_to_id))])
        for ngram in char_ngrams[i]:
            features[-1][ngram] += 1
        for ngram in word_ngrams[i]:
            features[-1][ngram] += 1
    
    # make id to label
    if id_to_label is None:
        id_to_label = {}
        for label in label_to_id:
            id_to_label[label_to_id[label]] = label
    
    return features, label_to_id, id_to_label


def train_model(char_n_max: int = 4, word_n_max: int = 1):
    """Train a Gaussian Naive Bayes classifier on the data."""

    # read data
    data = pd.read_csv("data/regdataset.csv")

    # make X (sentences) and y (labels)
    literary = data["transliterated"].tolist()
    colloquial = data["colloquial: annotator 1"].tolist() + data["colloquial: annotator 2"].tolist()
    X_raw = literary + colloquial
    y = (["literary"] * len(literary)) + (["colloquial"] * (len(colloquial)))

    # featurise
    X, label_to_id, id_to_label = featurise(X_raw, char_n_max=char_n_max, word_n_max=word_n_max)

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
    mean_diffs = gnb.theta_[0, :] - gnb.theta_[1, :]
    abs_mean_diffs = np.abs(mean_diffs)
    sorted_mean_diffs = np.argsort(abs_mean_diffs)[::-1]

    print("Most informative features (positive = colloquial):")
    for i in range(30):
        print(f"{id_to_label[sorted_mean_diffs[i]]:<20} {mean_diffs[sorted_mean_diffs[i]]:>8.4f}")
    
    # save model
    with open("models/model.pickle", "wb") as f:
        pickle.dump({
            "model": gnb,
            "label_to_id": dict(label_to_id),
            "id_to_label": dict(id_to_label),
            "char_n_max": char_n_max,
            "word_n_max": word_n_max
        }, f)

def load_model_and_test(path: str, X_raw: list[str]):
    """Load a model from a pickle file."""

    with open(path, "rb") as f:
        config = pickle.load(f)
    
    # load model stuff
    model = config["model"]
    label_to_id = config["label_to_id"]
    id_to_label = config["id_to_label"]
    char_n_max = config["char_n_max"]
    word_n_max = config["word_n_max"]

    # featurise
    X_test, _, _ = featurise(X_raw, char_n_max=char_n_max, word_n_max=word_n_max, label_to_id=label_to_id, id_to_label=id_to_label)
    for sent in X_test:
        for id in range(len(sent)):
            if sent[id] > 0:
                print(id_to_label[id], end=", ")
        print()

    # predict
    y_pred = model.predict(X_test)

    return y_pred

def main():
    train_model(char_n_max=4, word_n_max=0)

    test = [
        "changar mattrum ivargal taj mahalil thamizh puththagangalai padippaargal",
        "shankarum ivangalum taj mahalle tamil puthagangale padippaanga"
    ]
    results = load_model_and_test("models/model.pickle", test)
    print(results)


if __name__ == "__main__":
    main()