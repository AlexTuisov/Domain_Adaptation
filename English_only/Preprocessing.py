import io
import conllu
import pandas as pd
import numpy as np
from conll_df import conll_df
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from gensim.models.wrappers import FastText
import time

BATCH_SIZE = 20


def load_encodings(file_name):
    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def load_unordered_data(path_to_data):
    data_file = open(path_to_data, "r", encoding="utf-8")
    corpus = list(conllu.parse_incr(data_file))
    corpus_as_words_and_tags = []
    for sentence in corpus:
        words = []
        tags = []
        for entry in sentence:
            word = entry["form"]
            tag = entry["xpostag"]
            words.append(word)
            tags.append(tag)
        words = tuple(words)
        tags = tuple(tags)
        corpus_as_words_and_tags.append((words, tags))
    return corpus_as_words_and_tags


def load_data_as_pandas(path_to_data):
    df = pd.DataFrame(load_unordered_data(path_to_data))
    df.columns = ["words", "tags"]
    return df


def load_train_test_validation_sets(path_to_data):
    train_test, validation_set = train_test_split(load_data_as_pandas(path_to_data), test_size=0.2)
    return train_test, validation_set


class WordsDataset(Dataset):
    def __init__(self, dataframe, path_to_model, mode=None):
        self.data = dataframe
        self.mode = mode
        self.model = FastText.load_fasttext_format(path_to_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]


if __name__ == '__main__':
    t = time.time()
    path_to_data = "./ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-train.conllu"
    path_to_model = "cc.en.300.bin"
    x, y = load_train_test_validation_sets(path_to_data)
    my_dataset = WordsDataset(x, path_to_model)
    dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, num_workers=4)

    d = time.time()
    print(f"time elapsed: {d-t} seconds")

    for similar_word in my_dataset.model.similar_by_word("Avihay"):
        print("Word: {0}, Similarity: {1:.2}f".format(
            similar_word[0], similar_word[1]))
    p = time.time()
    print(f"time elapsed: {p-t} seconds")



