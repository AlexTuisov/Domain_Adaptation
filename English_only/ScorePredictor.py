import torch
import torch.nn as nn
import time
import math
import numpy as np
import Preprocessing as pr
import random
from gensim.models.wrappers import FastText
from sklearn.metrics import accuracy_score

PATH_TO_DATA = "train.conllu"
PATH_TO_MODEL = "cc.en.300.bin"
NUM_OF_LAYERS = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_tensor(row_of_words, row_of_tags, model):
    list_of_tensors = []
    for word in row_of_words:
        try:
            vector = model[word]
        except KeyError:
            print("have a key error there!")
            print(word)
        tensor = torch.from_numpy(vector).float()
        list_of_tensors.append(tensor)
    output_tensor = torch.cat(list_of_tensors, 0)
    print(output_tensor)
    output_tensor = output_tensor.view(len(list_of_tensors), 1, len(list_of_tensors[0]))

    return output_tensor


def scoring_function(self, state: np.array):
    return accuracy_score(self.encoded_tags, state)



class GRUPredictor(nn.Module):
    def __init__(self, words_size, tags_size, tags_embedding_size):
        super(GRUPredictor, self).__init__()
        self.words_size = words_size
        self.tags_size = tags_size
        self.tags_embedding_size = tags_embedding_size
        self.embedding = nn.Embedding(tags_size, tags_embedding_size)
        hidden_size = words_size + tags_embedding_size
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=NUM_OF_LAYERS, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.loss = nn.MSELoss()
        self.name = "my little net"

    def forward(self, embedded_words, tags, hidden):
        embedded_tags = self.embedding(tags)
        gru_input = torch.cat([embedded_words, embedded_tags], 0)
        output, hidden = self.gru(gru_input, hidden)
        output = self.linear(output[-1])
        output = self.loss(output)
        return output

    def initHidden(self, batch_size):
        return torch.zeros(NUM_OF_LAYERS, batch_size, self.hidden_size)


def train_one_epoch(epoch_number, train_dataset, optimizer, model, criterion):
    random_order = list(range(len(train_dataset)))
    random.shuffle(random_order)
    epoch_loss = 0
    row_counter = 0
    for i in random_order:
        batch = train_dataset[i]
        true_answers = [train_dataset[i][0][1]] * len(batch)
        true_scores = map(calculate_true_score, zip(batch, true_answers))
        output, loss = train_batch(optimizer, model, criterion, batch, true_scores)
        epoch_loss += loss



    average_loss = epoch_loss / row_counter
    print(f"This epoch the average loss was {average_loss:.4f}")
    return average_loss


def train_batch(optimizer, model, criterion, batch, true_scores):
    optimizer.zero_grad()
    hidden = model.initHidden(batch.size()[0]).to(DEVICE)
    output = model(batch, hidden)
    loss = criterion(output, true_scores)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
    optimizer.step()
    return output, loss.item()


def calculate_true_score(sentence_and_tags, true_tags):
    tags = sentence_and_tags[1]
    length = len(tags)
    assert length == len(true_tags)
    count = 0
    for i in range(length):
        if tags[i] == true_tags[i]:
            count += 1
    return count/length

if __name__ == '__main__':

    # my_model = GRUPredictor(300, 80, 8)
    # print(my_model.name)

    language_model = FastText.load_fasttext_format(PATH_TO_MODEL)
    x, y, tag_set = pr.load_train_test_validation_sets(PATH_TO_DATA)
    train_set = pr.WordsDataset(x, tag_set)
    test_set = pr.WordsDataset(y, tag_set)

    item = train_set[16][0]["words"]
    print(item)
    my_little_tensor = create_tensor(item, None, language_model)
    print(my_little_tensor.shape)