from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
import preprocessor as p
from nltk import RegexpTokenizer
import time


def confusion_matrix(y_true, y_pred):
    """
    Functie care creeaza matricea de confuzie pe un anume test
    :param y_true:
    :param y_pred:
    :return: conf_mx
    """
    unique_classes = len(np.unique(y_true))
    conf_mx = np.zeros((unique_classes, unique_classes))
    for i in range(0, len(y_pred)):
        conf_mx[int(y_true[i])][int(y_pred[i])] += 1
    return conf_mx


def tokenize(text):
    """Generic wrapper around different tokenization methods.
    """
    text = p.clean(text)
    tokens = RegexpTokenizer(r'\w+').tokenize(text)
    tokens = [word.lower() for word in tokens if len(word) > 3 and word.isalpha()]
    return tokens


def get_representation(vocabulary, how_many):
    """Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    wrd: @  che  .   ,   di  e
    idx: 0   1   2   3   4   5
    """
    most_comm = vocabulary.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for i, iterator in enumerate(most_comm):
        cuv = iterator[0]
        wd2idx[cuv] = i
        idx2wd[i] = cuv

    return wd2idx, idx2wd


def get_corpus_vocabulary(corpus):
    """Write a function to return all the words in a corpus.
    """
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter


def text_to_bow(text, wd2idx):
    """Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    """
    features = np.zeros(len(wd2idx))
    tokenz = tokenize(text)
    for tok in tokenz:
        if tok in wd2idx:
            features[wd2idx[tok]] += 1

    return features


def corpus_to_bow(corpus, wd2idx):
    """Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2

    """

    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))

    all_features = np.array(all_features)
    return all_features


def write_prediction(out_file, predictions):
    """A function to write the predictions to a file.
    """
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(int(pred)) + '\n'
            fout.write(linie)


def precision_recall_score(y_true_copy, y_pred_copy):
    """
    :param y_true_copy:
    :param y_pred_copy:
    :return: precision, recall
    """
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for idx in range(0, len(y_pred_copy)):
        if y_true_copy[idx] == 1:
            if y_pred_copy[idx] == 1:
                tp += 1
            else:
                fn += 1
        if y_true_copy[idx] == 0:
            if y_pred_copy[idx] == 0:
                tn += 1
            else:
                fp += 1
    precision = round(tp / (tp + fp), 2)
    recall = round(tp / (tp + fn), 2)
    return precision, recall


def F1_score(y_true, y_pred):
    """
    F1 score
    :param y_true:
    :param y_pred:
    :return:
    """
    precison, recall = precision_recall_score(y_true, y_pred)
    return 2 * (precison * recall / (precison + recall))


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']

toate_cuvintele = get_corpus_vocabulary(corpus)
# print(len(toate_cuvintele))
wd2idx, idx2wd = get_representation(toate_cuvintele, 5000)

data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label'].values

test_data = corpus_to_bow(test_df['text'], wd2idx)


scoruri = []
clf = MultinomialNB()
matrice_confuzie = np.zeros((2, 2))
start = time.time()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
for train, test in skf.split(data, labels):
    data_train, data_test = data[train], data[test]
    labels_train, labels_test = labels[train], labels[test]
    clf.fit(data_train, labels_train)
    skl_predictii = clf.predict(data_test)
    scor = F1_score(labels_test, skl_predictii)
    conf_mx = confusion_matrix(labels_test, skl_predictii)
    print(F1_score(labels_test, skl_predictii))
    matrice_confuzie += conf_mx
    scoruri.append(scor)

print(np.mean(scoruri), ' ', np.std(scoruri))
print(matrice_confuzie)

clf.fit(data, labels)
end = time.time()
predictie = clf.predict(test_data)
print(predictie)
write_prediction('predictii.csv', predictie)
print(end-start)
