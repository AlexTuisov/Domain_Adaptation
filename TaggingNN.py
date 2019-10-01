from typing import List
from Preprocessing import Sentence, ALL_TAGS
from sklearn.metrics import accuracy_score
import os
import numpy as np

class TaggingNN:
    """
    This class represents tagging network used to create initial tagging for given sentence
    """
    def __init__(self):
        self.model = None
        self.backup_path = 'trainNN_model.backup'
        self.trained = False

    def _load_backup(self):
        if os.path.exists(self.backup_path):
            # load
            self.trained = True
            pass

    def _save_backup(self):
        pass

    def train(self, sentences: List[Sentence]):
        # Given list of sentences do some magic, train model and save it
        for sentence in sentences:
            X = sentence.X
            # do some stuff
            continue
        self._save_backup()
        self.trained = True

    def test(self, test_sentences: List[Sentence]):
        # Given test sentences and trained model check accuracy of those initial taggings
        total_predicted = []
        total_real = []
        for sentence in test_sentences:
            predicted = self.predict(sentence.X)
            total_predicted += predicted
            total_real += sentence.Y
        print('Accuracy score is ', accuracy_score(total_real, total_predicted))

    def predict(self, sentence: Sentence) -> np.array:
        # Given sentence return list of initial taggings
        # Until Vitaliy implements proper NN I use random labeling based on true labeling
        result = sentence.Y
        for i in range(len(result)):
            if np.random.uniform() < 0.2:
                result[i] = np.random.randint(len(ALL_TAGS))
        return result
