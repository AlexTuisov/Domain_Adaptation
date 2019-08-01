import time
import random


import pandas as pd
from gensim.models.wrappers import FastText
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import conllu


BATCH_SIZE = 20
BOOTSTRAPPING_FACTOR = 2


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
    df, tag_set = load_unordered_data(path_to_data)
    df = pd.DataFrame(df)
    df.columns = ["words", "tags"]
    return df, tag_set


def load_train_test_validation_sets(path_to_data):
    my_data, tag_set = load_data_as_pandas(path_to_data)
    train_test, validation_set = train_test_split(my_data, test_size=0.2)
    return train_test, validation_set, tag_set


class WordsDataset(Dataset):
    def __init__(self, dataframe, tag_set, path_to_model, mode=None):
        self.tag_set = tag_set
        self.data = dataframe
        self.mode = mode
        self.model = FastText.load_fasttext_format(path_to_model)

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
            tags[index] = random.sample(self.tag_set, 1)[0]
        row_to_return[1] = tags
        return tuple(row_to_return)


if __name__ == '__main__':
    t = time.time()

    # a_file = open("/home/alex/Desktop/Python_projects/NLP/Optimization_algs/train.wtag", "r")
    path_to_data = "train.conllu"
    path_to_model = "cc.en.300.bin"
    x, y, tag_set = load_train_test_validation_sets(path_to_data)
    my_dataset = WordsDataset(x, tag_set, path_to_model)
    # dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, num_workers=4)

    for item in my_dataset[16]:
        print(item)



