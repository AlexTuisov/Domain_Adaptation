from TaggingNN import TaggingNN
from ScoringNN import ScoringNN
from English_only.Preprocessing import load_sentences, ALL_TAGS, INT_TO_TAG, Sentence, TAG_TO_INT
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
        self.taggingNN.load_backup()
        self.search_class_name = search_algorithm_name
        self.search_class = NAME_TO_CLASS[search_algorithm_name]
        self.search_params = search_algorithm_parameters
        self.scoringNN = ScoringNN()
        self.scoringNN.load_backup()

    def run(self, test_language='en', override=False):
        print(test_language, self.search_class_name, self.search_params)
        train_sentences = load_sentences('en', 'train', override)

        if not self.taggingNN.trained or override:
            self.taggingNN.train(train_sentences)

        if not self.scoringNN.trained or override:
            self.scoringNN.train(train_sentences)

        test_sentences = load_sentences(test_language, 'test', override)
        i = 0
        for sentence in train_sentences:
            if sentence.words.count('_') > 3:
                # print(sentence.words)
                i += 1
        print(len(train_sentences), i)
        pair_results = {x: 0 for x in product(ALL_TAGS, ALL_TAGS)}
        tag_results = {x: [0, 0] for x in ALL_TAGS}
        for sentence in test_sentences:
            start = time.time()
            init_state = self.taggingNN.predict(sentence)
            best_state = self.run_optimization_loop(sentence, init_state)
            for i, predicted_int in enumerate(best_state):
                predicted_tag = INT_TO_TAG[predicted_int]
                real_tag = sentence.tags[i]
                pair_results[(real_tag, predicted_tag)] += 1
                tag_results[real_tag][0] += real_tag == predicted_tag
                tag_results[real_tag][1] += 1
            print('Initial sentence accuracy:', round(accuracy_score(sentence.Y, init_state), 2), '. Final accuracy:',
                  round(accuracy_score(sentence.Y, best_state), 2), 'took', time.time() - start, 'seconds')

        self.save_results(tag_results, pair_results, test_language)

    def run_optimization_loop(self, sentence, init_state):
        fitness_function = mlrose.CustomFitness(partial(self.scoringNN.score, sentence))
        problem = mlrose.DiscreteOpt(length=len(sentence), fitness_fn=fitness_function, maximize=True,
                                     max_val=len(ALL_TAGS))

        try:
            best_state, best_fitness = self.search_class(problem, init_state=init_state, **self.search_params)
        except TypeError:
            best_state, best_fitness = self.search_class(problem, **self.search_params)
        return best_state

    def save_results(self, tag_results, pair_results, test_language='en'):
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
        with open('results/' + test_language + self.search_class_name + str(self.search_params).replace('\'', '') + '.json', 'w') as f:
            json.dump(pair_results, f)

    def check_performance(self, language='en'):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        list_of_sentences = load_sentences(language, 'test', False)
        results = []
        original_accuracies = []
        search_accuracies = []
        cnt = 0
        start = time.time()
        for sentence in list_of_sentences[:50]:
            res = []
            for num_errors in range(-1, 5):
                tags, true_tags, accuracy = self.scoringNN.mutate(sentence, num_errors)
                init_state = np.array([TAG_TO_INT[x] for x in tags])
                new_state = self.run_optimization_loop(sentence, init_state)
                new_num_errors = sum([TAG_TO_INT[true_tags[i]] != new_state[i] for i in range(len(new_state))])
                res.append(new_num_errors)
                original_accuracies.append(accuracy)
                search_accuracies.append(1-new_num_errors/len(sentence))

            results.append(res)
            cnt += 1
            if cnt%10 == 0:
                print(cnt, time.time()-start)
        sns.set()
        fig, ax = plt.subplots(5, 1, figsize=(8,30), sharex='all')
        df = pd.DataFrame(columns=['-1', '0', '1', '2', '3', '4'], data=results)
        for i in range(0,5):
            f = sns.barplot(x=str(i), y='-1', data=df[[str(i), '-1']].groupby(str(i)).count().reset_index(), ax=ax[i])
            f.set_title('Distribution of errors after optimization. Started with ' + str(i) +" errors")
            f.set_ylabel('Number of sentences')
            f.set_xlabel('')
            if i==4:
                f.set_xlabel('Number of errors')

        plt.savefig('errors_bars_' + language + '_' + self.search_class_name + '.png')
        #plt.show()

        sns.set()
        plt.figure(figsize=(12, 12))
        df = pd.DataFrame(columns=['original_accuracy', 'after_search_accuracy'], data=zip(original_accuracies, search_accuracies))
        sns.regplot(x='original_accuracy', y='after_search_accuracy', data=df)
        plt.ylim(0, 1.1)
        plt.xlim(0, 1.1)
        plt.title('Performance of scoring model with regression fit. ' + self.search_class_name)
        plt.savefig('errors_regression_' + language + '_' + self.search_class_name + '.png')
        plt.show()


if __name__ == "__main__":
    #algorithm_name = 'hill_climb'
    #algorithm_args = {'restarts': 0, 'max_iters': 100, 'random_state': 1}
    #algorithm_name = 'genetic'
    #algorithm_args = {'pop_size': 200, 'max_iters': 100}
    algorithm_name = 'random_hill_climb'
    algorithm_args = {'restarts': 1, 'max_iters': 4000, 'max_attempts': 100, 'random_state': 1}
    #train_sentences = load_sentences('fr', 'test', False)
    runner = Runner(algorithm_name, algorithm_args)
    # We kinda don't have validation in russian, so we need to tune model using english test set
    #runner.run('en', override=False)
    runner.check_performance('en')
