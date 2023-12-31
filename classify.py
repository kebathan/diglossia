"""
Train a Gaussian Naive Bayes classifier on distinguishing literary and colloquial Tamil.

Author: Aryaman Arora and Kabilan Prasanna
Date: 2023-09-10
"""

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import pickle
import variants
from tqdm import tqdm
import argparse
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
from datasets import load_dataset
import os

def load_data(train_on="regdata", test_on="both", augment=True):
    # augmentation functions
    fxs = [variants.ch_s, variants.gemination, variants.zh_l, variants.h_g, variants.le_la]

    X_train, y_train = [], []
    X_test, y_test = [], []

    # read data
    if train_on == "regdata" or train_on == "both":
        data = pd.read_csv("data/regdataset.csv")

        # make X (sentences) and y (labels)
        literary = data["transliterated"].tolist()
        colloquial = data["colloquial: annotator 1"].tolist() + data["colloquial: annotator 2"].tolist()

        # apply orthographical changes
        # print(len(literary), len(colloquial))
        if augment:
            for fx in fxs:
                lit = []
                for sent in literary:
                    changed = fx(sent)
                    if changed is not None:
                        lit.extend(changed) if isinstance(changed, list) else lit.append(changed)
                
                col = []
                for sent in colloquial:
                    changed = fx(sent)
                    if changed is not None:
                        col.extend(changed) if isinstance(changed, list) else col.append(changed)

                literary.extend(lit)
                colloquial.extend(col)

        # add to train
        X_raw = literary + colloquial
        y = (["literary"] * len(literary)) + (["colloquial"] * (len(colloquial)))
        X_train.extend(X_raw)
        y_train.extend(y)

    # no augmentation for dakshina
    if train_on == "both":
        literary = []
        with open("data/dakshina1.txt", "r") as data:
            literary.extend(data.readlines())
        with open("data/dakshina2.txt", "r") as data:
            literary.extend(data.readlines())
        X_train.extend(literary)
        y_train.extend(["literary"] * len(literary))
    
    # test only on dakshina
    if test_on == "dakshina":
        literary = []
        with open("data/dakshina1.txt", "r") as data:
            literary.extend(data.readlines())
        with open("data/dakshina2.txt", "r") as data:
            literary.extend(data.readlines())
        X_test.extend(literary)
        y_test.extend(["literary"] * len(literary))

    if test_on == "cc100":
        literary = []
        with open("data/cc100_head.txt", "r") as data:
            literary.extend(data.readlines())
        X_test.extend(literary)
        y_test.extend(["literary"] * len(literary))
    
    if test_on == "tamilmixsentiment":
        dataset = load_dataset('tamilmixsentiment')
        dataset = [row["text"] for split in dataset.keys() for row in dataset[split]]
        X_test.extend(dataset)
        y_test.extend(["literary"] * len(dataset))
    
    if test_on == "offenseval":
        dataset = load_dataset('offenseval_dravidian', 'tamil')
        dataset = [row["text"] for split in dataset.keys() for row in dataset[split]]
        X_test.extend(dataset)
        y_test.extend(["literary"] * len(dataset))
    
    if test_on == "hope_edi":
        dataset = load_dataset('hope_edi', 'tamil')
        dataset = [row["text"] for split in dataset.keys() for row in dataset[split]]
        X_test.extend(dataset)
        y_test.extend(["literary"] * len(dataset))
    
    if test_on == "both":
        # split train into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    # remove punctuation and lowercase
    X_train = [''.join([x for x in sent.lower() if x not in '.,\n']) for sent in X_train]
    X_test = [''.join([x for x in sent.lower() if x not in '.,\n']) for sent in X_test]

    return X_train, y_train, X_test, y_test

def finetune_xlm_roberta(
    train_on="both",
    test_on="both",
    lr=2e-5,
    epochs=4,
    augment=True,
    freeze=False,
    model_name="xlm-roberta-base",
    upload=False
):

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name, 
        num_labels = 2, 
        output_attentions = False,
        output_hidden_states = False
    ).to(device)

    # freeze
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    # read data
    X_train, y_train, X_test, y_test = load_data(train_on=train_on, test_on=test_on, augment=augment)

    # labels
    le = LabelEncoder()
    le.fit(y_train + y_test)

    # tokenize train
    tokenized_feature = tokenizer.batch_encode_plus(
        X_train, 
        add_special_tokens = True,
        padding = 'max_length',
        truncation=True,
        max_length = 128, 
        return_attention_mask = True,
        return_tensors = 'pt'       
    )
    train_inputs = tokenized_feature['input_ids']
    train_masks = tokenized_feature['attention_mask']
    train_labels = le.transform(y_train)

    # tokenize test
    tokenized_feature = tokenizer.batch_encode_plus(
        X_test, 
        add_special_tokens = True,
        padding = 'max_length',
        truncation=True,
        max_length = 128, 
        return_attention_mask = True,
        return_tensors = 'pt'       
    )
    validation_inputs = tokenized_feature['input_ids']
    validation_masks = tokenized_feature['attention_mask']
    validation_labels = le.transform(y_test)

    # define batch_size
    batch_size = 16

    # create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # create the DataLoader for our test set
    validation_data = TensorDataset(validation_inputs, validation_masks, torch.tensor(validation_labels))
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # optimiser
    optimizer = AdamW(model.parameters(),
        lr = lr, 
        eps = 1e-8 
    )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps)

    # training
    loss_values = []
    print('total steps per epoch: ',  len(train_dataloader) / batch_size)

    # looping over epochs
    for epoch_i in range(0, epochs):
        print('training on epoch: ', epoch_i)

        t0 = time.time()
        total_loss = 0
        model.train()

        # loop through batch
        for step, batch in enumerate(tqdm(train_dataloader)):
            # if step % 50 == 0 and not step == 0:
            #     print('training on step: ', step)
            #     print('total time used is: {0:.2f} s'.format(time.time() - t0))

            # load data from dataloader 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # clear any previously calculated gradients 
            model.zero_grad()

            # get outputs
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            
            # get loss + update
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("average training loss: {0:.2f}".format(avg_train_loss))

        t0 = time.time()
        # model in validation mode
        model.eval()
        # save prediction
        predictions, true_labels =[],[]
        # evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # validation
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)
            # get output
            logits = outputs[0]
            # move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            final_prediction = np.argmax(logits, axis=-1).flatten()
            predictions.append(final_prediction)
            true_labels.append(label_ids)
            
        # print('total time used is: {0:.2f} s'.format(time.time() - t0))

        # convert numeric label to string
        final_prediction_list = le.inverse_transform(np.concatenate(predictions))
        final_truelabel_list = le.inverse_transform(np.concatenate(true_labels))

        cr = classification_report(final_truelabel_list, 
                                final_prediction_list,
                                output_dict=True, zero_division=0)
        print(cr)

    # make directory if it doesn't exist
    os.makedirs("models/xlm_roberta", exist_ok=True)
    path = "models/xlm_roberta"
    model_path = f"{path}/model.pt"
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(path, push_to_hub=upload, repo_id=f"aryaman/{model_name}-irumozhi")
    model.save_pretrained(path, push_to_hub=upload, repo_id=f"aryaman/{model_name}-irumozhi")
    return cr

def featurise(
    sents,
    char_n_max: int = 3,
    word_n_max: int = 3,
    label_to_id = None,
    id_to_label = None
):
    """Featurise a list of sentences into a list of lists of n-grams."""

    # clean sents (remove punctuation, etc.)
    sents = [''.join([x for x in sent.lower().replace(' ', '_') if x not in '.,\n']) for sent in sents]

    # label to id and id to label
    provided_labels = True
    if label_to_id is None:
        provided_labels = False
        label_to_id = defaultdict(lambda: len(label_to_id))

    # make char n-grams
    # print("making char n-grams")
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
    # print("making word n-grams")
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
    # print("converting n-grams to counts")
    features = []
    for i in range(len(sents)):
        features.append(np.zeros(len(label_to_id)))
        for ngram in char_ngrams[i]:
            features[-1][ngram] += 1
        for ngram in word_ngrams[i]:
            features[-1][ngram] += 1
        # features[-1] /= features[-1].sum()
    
    # make id to label
    if id_to_label is None:
        id_to_label = {}
        for label in label_to_id:
            id_to_label[label_to_id[label]] = label
    
    return features, label_to_id, id_to_label

def train_model(
    model: str="gnb",
    train_on: str="both",
    test_on: str="both",
    char_n_max: int = 4,
    word_n_max: int = 1,
    augment: bool = True
):
    """Train a Gaussian Naive Bayes classifier on the data."""

    X_train, y_train, X_test, y_test = load_data(train_on, test_on, augment=augment)
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))

    # featurise
    X, label_to_id, id_to_label = featurise(X_train + X_test, char_n_max=char_n_max, word_n_max=word_n_max)
    X_train = X[:len(X_train)]
    X_test = X[len(X_train):]

    # create a Gaussian Naive Bayes classifier + train
    # print("starting model training")
    gnb = GaussianNB() if model == "gnb" else MultinomialNB()
    batch_size = 1000
    for i in range(0, len(X_train), batch_size):
        gnb.partial_fit(X_train[i:i+batch_size], y_train[i:i+batch_size], classes=["literary", "colloquial"])

    # predict
    # print("starting predictions")
    y_pred = []
    for i in range(0, len(X_test), batch_size):
        y_pred.extend(gnb.predict(X_test[i:i+batch_size]))
    y_pred = np.array(y_pred)

    # print results
    cr = classification_report(y_test, 
                            y_pred, 
                            output_dict=True, zero_division=0)
    print(cr)
    # print("Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))
    # print(Counter(y_pred))
    
    # print most informative features
    # if model == "gnb":
    #     mean_diffs = gnb.theta_[0, :] - gnb.theta_[1, :]
    #     abs_mean_diffs = np.abs(mean_diffs)
    #     sorted_mean_diffs = np.argsort(abs_mean_diffs)[::-1]

    #     print("Most informative features (positive = colloquial):")
    #     for i in range(5):
    #         print(f"{id_to_label[sorted_mean_diffs[i]]:<20} {mean_diffs[sorted_mean_diffs[i]]:>8.4f}")
    
    # save model
    with open("models/gnb.pickle", "wb") as f:
        pickle.dump({
            "model": gnb,
            "label_to_id": dict(label_to_id),
            "id_to_label": dict(id_to_label),
            "char_n_max": char_n_max,
            "word_n_max": word_n_max
        }, f)
    
    return gnb, y_test, y_pred, cr, X_train, y_train, X_test, y_test, label_to_id, id_to_label

def load_model_and_test(path: str, X_raw):
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

    # predict
    y_pred = []
    for i in tqdm(range(0, len(X_test), 1000)):
        y_pred.extend(model.predict(X_test[i:i+1000]))
    y_pred = np.array(y_pred)

    return y_pred

def test_files(files):
    test = []
    for f in files:
        with open(f, "r") as data:
            test.extend(data.readlines())
    
    return load_model_and_test("models/model.pickle", test)

def load_model_and_tokenize(model_path: str, tokenizer_path: str):

    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)

    model.eval()

    return model, tokenizer

def roberta_predict(sentences, model, tokenizer):
    
    predictions = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
        with torch.nograd():
            outputs = model(**inputs)

        logits = outputs.logits
        prediction = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(prediction).item()

        prediction.append(predicted_class)

    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnb', action='store_true', help='train Gaussian naive bayes model')
    parser.add_argument('--mnb', action='store_true', help='train Multinomial naive bayes model')
    parser.add_argument('--test_gnb', action='store_true', help='test model')
    parser.add_argument('--xlmr', action='store_true', help='finetune model')
    parser.add_argument('--no_augment', action='store_true', help='don\'t augment data')
    parser.add_argument('--char', type=int, default=4, help='max char n-gram')
    parser.add_argument('--word', type=int, default=1, help='max word n-gram')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=4, help='number of epochs')
    parser.add_argument('--train_on', type=str, default="both", help='train on dakshina or regdata')
    parser.add_argument('--test_on', type=str, default="both", help='test on dakshina or regdata')
    parser.add_argument('--freeze', action='store_true', help='freeze model')
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base", help='model name')
    parser.add_argument('--upload', action='store_true', help='upload model to huggingface')
    args = parser.parse_args()
    print(vars(args))

    if args.gnb or args.mnb:
        train_model(
            model="mnb" if args.mnb else "gnb",
            char_n_max=args.char,
            word_n_max=args.word,
            train_on=args.train_on,
            test_on=args.test_on,
            augment=not args.no_augment
        )

    if args.test_gnb:
        results = test_files(["data/dakshina2.txt"])
        print(Counter(results))
    
    if args.xlmr:
        finetune_xlm_roberta(
            lr=args.lr,
            epochs=args.epochs,
            train_on=args.train_on,
            test_on=args.test_on,
            augment=not args.no_augment,
            freeze=args.freeze,
            model_name=args.model_name,
            upload=args.upload
        )


if __name__ == "__main__":
    main()