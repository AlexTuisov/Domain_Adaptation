from typing import List
from English_only.Preprocessing import Sentence, INT_TO_TAG
import English_only.ScorePredictor as sc
from sklearn.metrics import accuracy_score
import os
import numpy as np
import torch
import torch.nn as nn
import random

NUMBER_OF_TAGS = len(INT_TO_TAG)
TAG_EMBEDDING_DIMENSION = 8
LEARNING_RATE = 0.001


class ScoringNN:
    def __init__(self):
        self.model = sc.GRUPredictor(sc.WORDS_EMBEDDING_DIMENSION, NUMBER_OF_TAGS, TAG_EMBEDDING_DIMENSION)
        self.criterion = None
        self.loss = torch.nn.MSELoss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.backup_path = 'trainNN_model.backup'
        self.trained = False

    def load_backup(self):
        if os.path.exists(self.backup_path):
            # load
            self.model.load_state_dict(torch.load(self.backup_path))
            self.model.eval()
            self.trained = True

    def save_backup(self):
        torch.save(self.model.state_dict(), self.backup_path)

    def train(self, sentences: List[Sentence]):
        for epoch in range(sc.NUMBER_OF_EPOCHS):
            random.shuffle(sentences)
            self.train_one_epoch(sentences)
            self.save_backup()
            self.trained = True

    def score(self, sentence: Sentence, tags: np.array) -> float:
        # return score of given sentence with given tags
        return accuracy_score(sentence.Y, tags)

    def train_one_epoch(self, sentences):
        cumulative_loss = 0
        for sentence in sentences:
            length_of_sentence = len(sentence)
            sentence_as_tensor = torch.LongTensor(sentence.X).unsqueeze(0).to(sc.DEVICE)
            # tags_as_np_array = np.array(sentence.Y)
            # true_tags_as_tensor = np.zeros(length_of_sentence, NUMBER_OF_TAGS)
            # true_tags_as_tensor[np.arange(length_of_sentence), tags_as_np_array] = 1
            # true_tags_as_tensor = torch.LongTensor(true_tags_as_tensor).unsqueeze(0).to(sc.DEVICE)
            output, loss = self.train_one_example(sentence_as_tensor, sentence.Y)

    def train_one_example(self, sentence_as_tensor, true_tags):
        true_accuracy = None
        hidden = self.model.initHidden(sentence_as_tensor.size()[0]).to(sc.DEVICE)
        output = self.model(sentence_as_tensor, hidden)

        return None, None

if __name__ == '__main__':
    my_test_scoring_nn = ScoringNN()