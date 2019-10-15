import time
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import conllu
from typing import List
import os
import gensim
import pickle

BATCH_SIZE = 20
BOOTSTRAPPING_FACTOR = 2
ALL_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X']
ROOT_PATH = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
TAG_TO_INT = {tag: i for i, tag in enumerate(ALL_TAGS)}
INT_TO_TAG = {v: k for k, v in TAG_TO_INT.items()}


class Sentence:
    def __init__(self, words, tags, encoder):
        self.words = words
        self.tags = tags
        self.X = []
        self.Y = []
        for word in words:
            word = word.lower()
            if word in encoder.vocab:
                self.X.append(encoder[word])
            else:
                return
        self.Y = np.array([TAG_TO_INT[x] for x in tags])

    def __len__(self):
        return len(self.words)

    def encode(self, encoder):
        self.X = []
        for word in self.words:
            word = word.lower()
            self.X.append(encoder[word])


class TrainSentence(Sentence):
    pass


def load_unordered_data(path_to_data):
    data_file = open(path_to_data, "r", encoding="utf-8")
    corpus = list(conllu.parse_incr(data_file))
    corpus_as_words_and_tags = []
    for sentence in corpus:
        words = []
        tags = []
        for entry in sentence:
            word = entry["form"]
            tag = entry["upostag"]
            if tag not in ALL_TAGS:
                words = []
                break
            words.append(word)
            tags.append(tag)
        if words:
            words = tuple(words)
            tags = tuple(tags)
            corpus_as_words_and_tags.append((words, tags))
    return corpus_as_words_and_tags


def load_data_as_pandas(path_to_data):
    df = load_unordered_data(path_to_data)
    df = pd.DataFrame(df)
    df.columns = ["words", "tags"]
    return df


def load_train_test_validation_sets(path_to_data):
    my_data = load_data_as_pandas(path_to_data)
    train_test, validation_set = train_test_split(my_data, test_size=0.2, random_state=1)
    return train_test, validation_set


def load_sentences(language='en', action='train', override=False) -> List[Sentence]:
    path = os.path.join(ROOT_PATH, 'data', language+'_'+action)
    if not os.path.exists(path+'_encoded.pkl') or override:
        with open(path+'.pkl', 'rb') as f:
            sentences = pickle.load(f)

        print('Loading fasttext encoder for', language)
        encoder = gensim.models.KeyedVectors.load_word2vec_format("./wiki."+language+".align.vec")
        print('Finished loading')

        for s in sentences:
            s.encode(encoder)

        with open(path+'_encoded.pkl', 'wb') as f:
            pickle.dump(sentences, f)
        return sentences

    with open(path + '_encoded.pkl', 'rb') as f:
        return pickle.load(f)


def filter_all_data(data_path=os.path.join(ROOT_PATH, 'ud-treebanks'), language_full='English', language_short='en',
                    train_size=0.8):
    corpus_as_words_and_tags = []
    for foldername in os.listdir(data_path):
        if not foldername.startswith('UD_'+language_full):
            continue
        for filename in os.listdir(os.path.join(data_path, foldername)):
            if not filename.endswith('.conllu'):
                continue
            corpus_as_words_and_tags += load_unordered_data(os.path.join(data_path, foldername, filename))
    print('Total sentences: ', len(corpus_as_words_and_tags))
    print('Loading fasttext encoder for', language_full)
    encoder = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(ROOT_PATH, "wiki." + language_short + ".align.vec"))
    print('Finished loading')
    result = []
    for words, tags in corpus_as_words_and_tags:
        sent = Sentence(words, tags, encoder)
        if len(sent.Y):
            result.append(sent)
    print('Total sentences after filtering: ', len(result))
    del encoder
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(result, train_size=train_size)
    del result
    time.sleep(3)
    if language_short == 'en':
        with open(os.path.join(ROOT_PATH, 'data', language_short+'_train_encoded.pkl'), 'wb') as f:
            pickle.dump(train, f)
        for s in train:
            s.X = []
        with open(os.path.join(ROOT_PATH, 'data', language_short+'_train.pkl'), 'wb') as f:
            pickle.dump(train, f)

    with open(os.path.join(ROOT_PATH, 'data', language_short+'_test_encoded.pkl'), 'wb') as f:
        pickle.dump(test, f)
    for s in test:
        s.X = []
    with open(os.path.join(ROOT_PATH, 'data', language_short+'_test.pkl'), 'wb') as f:
        pickle.dump(test, f)


class WordsDataset(Dataset):
    def __init__(self, dataframe,mode=None):
        self.data = dataframe
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.bootstrap_data(self.data.iloc[index])

    def bootstrap_data(self, row_to_be_bootstrapped):
        output = []
        for _ in range(BOOTSTRAPPING_FACTOR):
            for num_of_errors in range(len(row_to_be_bootstrapped[0])):
                output.append(self.change_row(row_to_be_bootstrapped, num_of_errors))
        return output

    def change_row(self, original_row, num_of_errors):
        # original_row = tuple(original_row)
        if num_of_errors <= 0:
            return original_row
        indexes = random.sample(list(range(len(original_row[0]))), num_of_errors)
        row_to_return = list(original_row)
        tags = list(row_to_return[1])
        for index in indexes:
            tags[index] = random.sample(ALL_TAGS, 1)[0]
        row_to_return[1] = tags
        return tuple(row_to_return)


if __name__ == '__main__':

    path_to_data = "train.conllu"
    path_to_model = "cc.en.300.bin"
    x, y = load_train_test_validation_sets(path_to_data)
    my_dataset = WordsDataset(x)

    for item in my_dataset[16]:
        print(item)



