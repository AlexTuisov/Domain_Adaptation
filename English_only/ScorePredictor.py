import torch
import torch.nn as nn
import time
import math
import numpy as np
import Preprocessing as pr
from sklearn.metrics import accuracy_score

PATH_TO_DATA = "train.conllu"
PATH_TO_MODEL = "cc.en.300.bin"
NUM_OF_LAYERS = 2

if __name__ == '__main__':
    x, y, tag_set = pr.load_train_test_validation_sets(PATH_TO_DATA)
    train_set = pr.WordsDataset(x, tag_set, PATH_TO_MODEL)
    test_set = pr.WordsDataset(y, tag_set, PATH_TO_MODEL)

    for item in train_set[16]:
        print(item)


def create_tensor(row_of_words, row_of_tags):
    pass


def scoring_function(self, state: np.array):
    return accuracy_score(self.encoded_tags, state)



class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=NUM_OF_LAYERS, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = self.linear(output[-1])
        output = self.softmax(output)
        return output

    def initHidden(self, batch_size):
        return torch.zeros(NUM_OF_LAYERS, batch_size, self.hidden_size)


