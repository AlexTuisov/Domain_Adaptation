from typing import List
from Preprocessing import Sentence
from sklearn.metrics import accuracy_score
import os
from numpy import array


class ScoringNN:
    def __init__(self):
        self.model = None
        self.backup_path = 'trainNN_model.backup'
        self.trained = False

    def load_backup(self):
        if os.path.exists(self.backup_path):
            # load
            self.trained = True
            pass

    def save_backup(self):
        pass

    def train(self, sentences: List[Sentence]):
        for sentence in sentences:
            X = sentence.X
            # do some stuff
            continue
        self.save_backup()
        self.trained = True

    def score(self, sentence: Sentence, tags: array) -> float:
        # return score of given sentence with given tags
        return accuracy_score(sentence.Y, tags)
