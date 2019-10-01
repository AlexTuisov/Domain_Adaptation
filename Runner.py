from TaggingNN import TaggingNN
from ScoringNN import ScoringNN
from English_only.Preprocessing import load_sentences, ALL_TAGS, INT_TO_TAG
import mlrose
from functools import partial
from itertools import product
import os
import pickle
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
            fitness_function = mlrose.CustomFitness(partial(self.scoringNN.score, sentence))
            problem = mlrose.DiscreteOpt(length=len(sentence), fitness_fn=fitness_function, maximize=True,
                                         max_val=len(ALL_TAGS))

            init_state = self.taggingNN.predict(sentence)

            best_state, best_fitness = self.search_class(problem, init_state=init_state, **self.search_params)
            for i, predicted_int in enumerate(best_state):
                predicted_tag = INT_TO_TAG[predicted_int]
                real_tag = sentence.tags[i]
                pair_results[(real_tag, predicted_tag)] += 1
                tag_results[real_tag][0] += real_tag == predicted_tag
                tag_results[real_tag][1] += 1

        os.makedirs('results', exist_ok=True)
        for tag in ALL_TAGS:
            if tag_results[tag][1] == 0:
                continue
            print('Accuracy for tag', tag, 'is:', tag_results[tag][0]/tag_results[tag][1])

        with open('results/' + self.search_class_name+ '.pkl', 'wb') as f:
            pickle.dump(pair_results, f)


if __name__ == "__main__":
    algorithm_name = 'hill_climb'
    algorithm_args = {'restarts': 0, 'max_iters': 1000, 'random_state': 1}
    #train_sentences = load_sentences('fr', 'test', False)
    runner = Runner(algorithm_name, algorithm_args)
    runner.run('en')
