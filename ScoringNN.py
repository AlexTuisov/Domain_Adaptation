from typing import List
from English_only.Preprocessing import Sentence, TAG_TO_INT, ALL_TAGS
from sklearn.metrics import accuracy_score
import os
import numpy as np
import torch
import torch.nn as nn
import random

NUMBER_OF_TAGS = len(ALL_TAGS)
TAG_EMBEDDING_DIMENSION = 8
LEARNING_RATE = 0.001
NUM_OF_LAYERS = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WORDS_EMBEDDING_DIMENSION = 300
NUMBER_OF_EPOCHS = 10


class ScoringNN:
    def __init__(self):
        self.criterion = torch.nn.MSELoss()
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

    def save_backup(self):
        torch.save(self.model.state_dict(), self.backup_path)

    def train(self, sentences: List[Sentence]):
        self.model.to(DEVICE)
        for epoch in range(NUMBER_OF_EPOCHS):
            random.shuffle(sentences)
            self.train_one_epoch(sentences)
            self.save_backup()
            self.trained = True

    def train_one_epoch(self, sentences):
        cumulative_loss = 0
        for sentence in sentences:
            length_of_sentence = len(sentence)
            output, loss = self.train_one_example(sentence)
        print("done AN EPOCH")

    def train_one_example(self, sentence):
        if random.random() < 0.0001:
            print("training an example!")
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
        return output, loss.item()

    def score(self, sentence: Sentence, tags: np.array) -> float:
        # return score of given sentence with given tags
        return accuracy_score(sentence.Y, tags)

    def mutate(self, sentence: Sentence):
        true_tags = sentence.tags
        num_of_errors = random.choice(range(len(sentence.tags) + 1))
        indexes = random.sample(list(range(len(sentence.tags))), num_of_errors)
        tags = list(true_tags)
        for index in indexes:
            tags[index] = random.sample(ALL_TAGS, 1)[0]
        correct_tags_count = 0
        for i, tag in enumerate(tags):
            if tag == true_tags[i]:
                correct_tags_count += 1
        accuracy = correct_tags_count / len(tags)
        return tags, true_tags, accuracy

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
        self.hidden_size = words_size + tags_embedding_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=NUM_OF_LAYERS, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, 1)
        self.name = "my little net"

    def forward(self, embedded_word, tag, hidden):
        tag = tag.type(torch.LongTensor).to(DEVICE)
        embedded_tags = self.embedding(tag)
        gru_input = torch.cat([embedded_word, embedded_tags], 2)
        output, hidden = self.gru(gru_input, hidden)
        output = self.linear(output[-1])
        return output

    def initHidden(self, batch_size):
        return torch.zeros(NUM_OF_LAYERS, batch_size, self.hidden_size).to(DEVICE)



if __name__ == '__main__':
    my_test_scoring_nn = ScoringNN()