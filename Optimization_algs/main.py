import mlrose
import numpy as np
from sklearn.metrics import accuracy_score
from functools import partialmethod
POS_to_int = {}

class Sentence:
    def __init__(self, line: str):
        """
        :param line: line from file in format %word_%POS
        """
        pairs = line.split()
        self.words = []
        self.tags = []
        for pair in pairs:
            word, pos = pair.split('_')
            # this is to ignore punctuation
            pos = ''.join([x for x in pos if x.isalpha()])
            if len(pos):
                self.words.append(word)
                self.tags.append(pos)
            if pos not in POS_to_int.keys():
                POS_to_int[pos] = len(POS_to_int)
        self.encoded_tags = np.array([POS_to_int[x] for x in self.tags])
        self.length = len(self.tags)

    def scoring_function(self, state: np.array):
        return accuracy_score(self.encoded_tags, state)

if __name__ == "__main__":
    sentences = []
    with open('train.wtag', 'r') as f:
        string = f.readline()
        while string:
            s = Sentence(string)
            sentences.append(s)
            string = f.readline()

    for i in range(10):
        test_sentence = sentences[i]
        # fitness function is function, receiving state and returning score.
        # State is np.array of integers between 0 and max_val-1, which is amount of POS tags in our case
        fitness_function = mlrose.CustomFitness(test_sentence.scoring_function)
        problem = mlrose.DiscreteOpt(length=test_sentence.length, fitness_fn=fitness_function, maximize=True,
                                     max_val=len(POS_to_int))

        # Define initial state
        init_state = np.zeros(len(test_sentence.tags))

        # Solve problem using simulated annealing
        best_state, best_fitness1 = mlrose.simulated_annealing(problem, max_attempts=10, max_iters=1000,
                                                              init_state=init_state, random_state=1)

        best_state, best_fitness2 = mlrose.hill_climb(problem, restarts=0, max_iters=1000,
                                                     init_state=init_state, random_state=1)

        best_state, best_fitness3 = mlrose.random_hill_climb(problem, max_attempts=10, max_iters=1000,
                                                            init_state=init_state, random_state=1)

        best_state, best_fitness4 = mlrose.genetic_alg(problem, max_attempts=10, max_iters=1000, random_state=1)

        best_state, best_fitness5 = mlrose.mimic(problem, max_attempts=10, max_iters=1000, random_state=1)

        print(i, ':\nannealing:', best_fitness1, '\nhill climb:', best_fitness2, '\nrandom hill climb:', best_fitness3,
              '\ngenetic', best_fitness4, '\nmimic', best_fitness5, '\n')
