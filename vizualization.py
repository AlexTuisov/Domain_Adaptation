import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
from ScoringNN import DEVICE, ScoringNN
from Preprocessing import load_sentences, TAG_TO_INT, Sentence



def plot_accuracies(language='en'):
    my_test_scoring_nn = ScoringNN()
    my_test_scoring_nn.load_backup()
    my_test_scoring_nn.model.to(DEVICE)
    list_of_sentences = load_sentences(language, 'test', False)
    start = time.time()
    accuracies = []
    predictions = []
    for sentence in list_of_sentences[:500]:
        for i in range(10):
            tags, true_tags, accuracy = my_test_scoring_nn.mutate(sentence)
            prediction = my_test_scoring_nn.score(sentence, [TAG_TO_INT[x] for x in tags])
            accuracies.append(accuracy)
            predictions.append(prediction)
    print(len(accuracies), 'took', time.time()-start, 'seconds')

    plt.figure(figsize=(12, 12))
    sns.set()
    df = pd.DataFrame(columns=['real_accuracy', 'predicted_value'], data=zip(accuracies, predictions))
    df['error'] = df.predicted_value - df.real_accuracy
    df['rounded_accuracy'] = df.real_accuracy.round(1)

    sns.regplot(x='real_accuracy', y='predicted_value', data=df, x_bins=20, ci=99)
    plt.ylim(0,1)
    #sns.scatterplot(x='real_accuracy', y='predicted_value', data=df, size=3)
    plt.title('Performance of scoring NN(' + language +'). 99% confidence interval')
    plt.savefig('regplot_'+language+'_.png')
    #plt.show()

    plt.figure(figsize=(12, 12))
    sns.set()
    sns.boxplot(x='rounded_accuracy', y='error', data=df)
    plt.title('Error of scoring NN by accuracy(' +language +').')
    plt.savefig('boxplot_' + language + '_.png')
    #plt.show()

    #print(f"cumulative error was {cumulative_error}, average test error was {cumulative_error/len(list_of_sentences)}")

def check_monotonity(language='en'):
    my_test_scoring_nn = ScoringNN()
    my_test_scoring_nn.load_backup()
    my_test_scoring_nn.model.to(DEVICE)
    list_of_sentences = load_sentences(language, 'test', False)
    #print(np.mean([len(x) for x in list_of_sentences]))
    for sentence in list_of_sentences[:100]:
        errors = [0]
        predictions = [my_test_scoring_nn.score(sentence, sentence.Y)]

        for num_errors in range(1, int(len(sentence)/2)):
            for i in range(20):
                tags, true_tags, accuracy = my_test_scoring_nn.mutate(sentence, num_errors)
                prediction = my_test_scoring_nn.score(sentence, [TAG_TO_INT[x] for x in tags])
                errors.append(num_errors)
                predictions.append(prediction)
        df = pd.DataFrame(columns=['num_errors', 'predicted_value'], data=zip(errors, predictions))
        sns.catplot(x='num_errors', y='predicted_value', data=df)
        plt.show()



if __name__ == '__main__':
    for lang in ['en', 'ru']:
        plot_accuracies(lang)
    #check_monotonity('en')
