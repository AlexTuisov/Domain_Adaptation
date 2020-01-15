import time
from typing import List
from English_only.Preprocessing import Sentence, TAG_TO_INT, INT_TO_TAG, ALL_TAGS, load_sentences
from sklearn.metrics import accuracy_score
import os
import numpy as np
import torch
import torch.nn as nn
import random

import pandas as pd
NUMBER_OF_TAGS = len(ALL_TAGS)
TAG_EMBEDDING_DIMENSION = 4
LEARNING_RATE = 0.0001
NUM_OF_LAYERS = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WORDS_EMBEDDING_DIMENSION = 300
MAX_NUMBER_OF_EPOCHS = 500
LARGE_INT = 9999999999
STOP_TRAINING_CRITERION = 3
HIDDEN_LAYER_SIZE = 304
BIDIRECTIONAL = True
MUTATIONS = 1


class ScoringNN:
    def __init__(self):
        self.criterion = torch.nn.L1Loss()
        self.model = GRUPredictor(WORDS_EMBEDDING_DIMENSION, NUMBER_OF_TAGS, TAG_EMBEDDING_DIMENSION)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.backup_path = 'trainNN_model.backup'
        self.trained = False

    def load_backup(self):
        if os.path.exists(self.backup_path):
            # load
            self.model.load_state_dict(torch.load(self.backup_path))
            self.model.eval()
            self.trained = True
            self.model.to(DEVICE)

    def save_backup(self):
        torch.save(self.model.state_dict(), self.backup_path)

    def train(self, sentences: List[Sentence]):
        self.model = GRUPredictor(WORDS_EMBEDDING_DIMENSION, NUMBER_OF_TAGS, TAG_EMBEDDING_DIMENSION)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.model.to(DEVICE)
        self.model.train()
        best_loss = LARGE_INT
        count_of_fails = 0
        for epoch in range(MAX_NUMBER_OF_EPOCHS):
            print(f"starting epoch {epoch}")
            random.shuffle(sentences)
            loss = self.train_one_epoch(sentences)
            if loss < best_loss:
                best_loss = loss
                count_of_fails = 0
                self.save_backup()
            else:
                count_of_fails += 1
            if count_of_fails >= STOP_TRAINING_CRITERION:
                print(f"stopped at epoch {epoch}")
                break
        self.trained = True

    def train_one_epoch(self, sentences):
        start = time.time()
        cumulative_loss = 0
        for sentence in sentences:
            loss = self.train_one_example(sentence)
            cumulative_loss += loss
        finish = time.time()
        print(f"finished, cumulative loss is {cumulative_loss},"
              f" average loss is {cumulative_loss/len(sentences)},"
              f"  took {finish - start} seconds")
        print("-----------------------")
        return cumulative_loss

    def train_one_example(self, sentence):
        self.optimizer.zero_grad()
        tags, true_tags, accuracy = self.mutate(sentence)
        sentence_as_tensor = torch.FloatTensor(sentence.X).unsqueeze(0).to(DEVICE)
        tags_as_tensor = self.tags_to_tensor(tags).unsqueeze(0).to(DEVICE)
        hidden = self.model.initHidden(sentence_as_tensor.size()[0]).to(DEVICE)
        output = self.model(sentence_as_tensor, tags_as_tensor, hidden)
        output = output[-1]
        output = output.unsqueeze(0)
        accuracy_as_tensor = torch.Tensor([accuracy]).unsqueeze(0).to(DEVICE)
        loss = self.criterion(output, accuracy_as_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
        self.optimizer.step()
        return loss.item()

    def score(self, sentence: Sentence, tags: List[int]) -> float:
        tags_as_str = [INT_TO_TAG[x] for x in tags]
        # return score of given sentence with given tags
        sentence_as_tensor = torch.FloatTensor(sentence.X).unsqueeze(0).to(DEVICE)
        tags_as_tensor = self.tags_to_tensor(tags_as_str).unsqueeze(0).to(DEVICE)
        hidden = self.model.initHidden(sentence_as_tensor.size()[0]).to(DEVICE)
        output = self.model(sentence_as_tensor, tags_as_tensor, hidden)
        output = output[-1].item()
        return output

    def mutate(self, sentence: Sentence, num_of_errors=-1):
        true_tags = sentence.tags
        if num_of_errors == -1:
            num_of_errors = random.choice(range(len(sentence.tags) + 1))
        else:
            num_of_errors = min(len(sentence), num_of_errors)
        indexes = random.sample(list(range(len(sentence.tags))), num_of_errors)
        tags = list(true_tags)
        for index in indexes:
            tags[index] = random.sample(ALL_TAGS, 1)[0]
        return tags, true_tags, accuracy_score(true_tags, tags)

    def tags_to_tensor(self, tags: List[str]) -> torch.Tensor:
        a = [TAG_TO_INT[x] for x in tags]
        a = torch.FloatTensor(a)
        return a


class GRUPredictor(nn.Module):
    def __init__(self, words_size, tags_size, tags_embedding_size):
        super(GRUPredictor, self).__init__()
        self.words_size = words_size
        self.tags_size = tags_size
        self.tags_embedding_size = tags_embedding_size
        self.embedding = nn.Embedding(tags_size, tags_embedding_size)
        self.input_size = words_size + tags_embedding_size
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=HIDDEN_LAYER_SIZE, num_layers=NUM_OF_LAYERS,
                          batch_first=True, bidirectional=BIDIRECTIONAL)
        self.linear = nn.Linear((int(BIDIRECTIONAL)+1) * HIDDEN_LAYER_SIZE, 1)
        self.sigmoid = nn.Sigmoid()
        self.name = "my little net"

    def forward(self, embedded_sentence, tag, hidden):
        tag = tag.type(torch.LongTensor).to(DEVICE)
        embedded_tags = self.embedding(tag)
        gru_input = torch.cat([embedded_sentence, embedded_tags], 2)
        output, hidden = self.gru(gru_input, hidden)
        output = self.linear(output[-1])
        output = self.sigmoid(output)
        return output

    def initHidden(self, batch_size):
        return torch.zeros((int(BIDIRECTIONAL) + 1) * NUM_OF_LAYERS, batch_size, HIDDEN_LAYER_SIZE).to(DEVICE)

