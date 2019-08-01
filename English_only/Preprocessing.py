import io
import time
import conllu
import pandas as pd
from gensim.models.wrappers import FastText
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

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
    tag_set = set()
    corpus = list(conllu.parse_incr(data_file))
    corpus_as_words_and_tags = []
    for sentence in corpus:
        words = []
        tags = []
        for entry in sentence:
            word = entry["form"]
            tag = entry["xpostag"]
            tag_set.add(tag)
            words.append(word)
            tags.append(tag)
        words = tuple(words)
        tags = tuple(tags)
        corpus_as_words_and_tags.append((words, tags))
    return corpus_as_words_and_tags, tag_set


def load_data_as_pandas(path_to_data):
    df, tag_set = pd.DataFrame(load_unordered_data(path_to_data))
    df.columns = ["words", "tags"]
    return df, tag_set


def load_train_test_validation_sets(path_to_data):
    train_test, validation_set = train_test_split(load_data_as_pandas(path_to_data), test_size=0.2)
    return train_test, validation_set


class WordsDataset(Dataset):
    def __init__(self, dataframe, tag_set, path_to_model, mode=None):
        self.tag_set = tag_set
        self.data = dataframe
        self.mode = mode
        self.model = FastText.load_fasttext_format(path_to_model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index, bootstrapped=False):
        return self.data.iloc[index]

    def bootstrap_data(self, row_to_be_bootstrapped):
        output = []
        for num_of_errors in range(len(row_to_be_bootstrapped)):
            output.append(self.change_row(row_to_be_bootstrapped, num_of_errors))
        return output

    def change_row(self, original_row, num_of_errors):
        #TODO: implement
        pass



if __name__ == '__main__':
    t = time.time()
    path_to_data = "./ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-train.conllu"
    path_to_model = "cc.en.300.bin"
    x, y = load_train_test_validation_sets(path_to_data)
    my_dataset = WordsDataset(x, path_to_model)
    dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, num_workers=4)

    d = time.time()
    print(f"time elapsed: {d-t} seconds")

    for similar_word in my_dataset.model.similar_by_word("cat"):
        print("Word: {0}, Similarity: {1:.2}".format(
            similar_word[0], similar_word[1]))
    p = time.time()
    print(f"time elapsed: {p-t} seconds")



