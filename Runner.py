from TaggingNN import TaggingNN
from ScoringNN import ScoringNN
from English_only.Preprocessing import load_sentences, ALL_TAGS, INT_TO_TAG, Sentence
import mlrose
from functools import partial
from itertools import product
import os
from sklearn.metrics import accuracy_score
import json
import time
"""
This file should be main platform for experiments
"""

NAME_TO_CLASS = {'annealing': mlrose.simulated_annealing, 'hill_climb': mlrose.hill_climb,
                 'genetic': mlrose.genetic_alg, 'random_hill_climb': mlrose.random_hill_climb}


class Runner:
    def __init__(self, search_algorithm_name, search_algorithm_parameters):
        # X_train, X_test, tag_set = load_train_test_validation_sets(test_lagnuage)
        self.taggingNN = TaggingNN()
        self.search_class_name = search_algorithm_name
        self.search_class = NAME_TO_CLASS[search_algorithm_name]
        self.search_params = search_algorithm_parameters
        self.scoringNN = ScoringNN()

    def run(self, test_language='en', override=False):
        print(test_language, self.search_class_name, self.search_params)
        if not self.taggingNN.trained or not self.scoringNN.trained or override:
            train_sentences = load_sentences('en', 'train', override)
            if not self.taggingNN.trained or override:
                self.taggingNN.train(train_sentences)
            if not self.scoringNN.trained or override:
                self.scoringNN.train(train_sentences)

        test_sentences = load_sentences(test_language, 'test', override)
        pair_results = {x: 0 for x in product(ALL_TAGS, ALL_TAGS)}
        tag_results = {x: [0, 0] for x in ALL_TAGS}
        for sentence in test_sentences:
            start = time.time()
            fitness_function = mlrose.CustomFitness(partial(self.scoringNN.score, sentence))
            problem = mlrose.DiscreteOpt(length=len(sentence), fitness_fn=fitness_function, maximize=True,
                                         max_val=len(ALL_TAGS))

            init_state = self.taggingNN.predict(sentence)
            try:
                best_state, best_fitness = self.search_class(problem, init_state=init_state, **self.search_params)
            except TypeError:
                best_state, best_fitness = self.search_class(problem, **self.search_params)
            for i, predicted_int in enumerate(best_state):
                predicted_tag = INT_TO_TAG[predicted_int]
                real_tag = sentence.tags[i]
                pair_results[(real_tag, predicted_tag)] += 1
                tag_results[real_tag][0] += real_tag == predicted_tag
                tag_results[real_tag][1] += 1
            print('Initial sentence accuracy:', round(accuracy_score(sentence.Y, init_state),2), '. Final accuracy:',
                  round(accuracy_score(sentence.Y, best_state),2), 'took', time.time()-start, 'seconds')
        hits = 0
        total = 0
        for tag in ALL_TAGS:
            if tag_results[tag][1] == 0:
                continue
            print('Accuracy for tag', tag, 'is:', tag_results[tag][0]/tag_results[tag][1])
            hits += tag_results[tag][0]
            total += tag_results[tag][1]
        print('Total accuracy: ', hits/total)
        os.makedirs('results', exist_ok=True)
        with open('results/' + self.search_class_name + str(self.search_params).replace('\'', '') + '.json', 'w') as f:
            json.dump(pair_results, f)


if __name__ == "__main__":
    algorithm_name = 'hill_climb'
    algorithm_args = {'restarts': 0, 'max_iters': 100, 'random_state': 1}
    #algorithm_name = 'genetic'
    #algorithm_args = {'pop_size': 200, 'max_iters': 100}
    #algorithm_name = 'random_hill_climb'
    #algorithm_args = {'restarts': 1, 'max_iters': 100000, 'max_attempts': 100, 'random_state': 1}
    #train_sentences = load_sentences('fr', 'test', False)
    runner = Runner(algorithm_name, algorithm_args)
    # We kinda don't have validation in russian, so we need to tune model using english test set
    runner.run('en')
