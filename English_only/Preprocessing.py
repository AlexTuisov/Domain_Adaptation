import io
import glob
import os
import conllu
import random
from pathlib import Path


def load_encodings(file_name):
    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def load_unordered_data(path_to_data):
    data_file = open(path_to_data, "r", encoding="utf-8")
    corpus = list(conllu.parse_incr(data_file))
    corpus_as_words_and_tags = {}
    for sentence in corpus:
        words = []
        tags = []
        for entry in sentence:
            word = entry["form"]
            tag = entry["xpostag"]
            words.append(word)
            tags.append(tag)
        words = tuple(words)
        tags = tuple(tags)
        corpus_as_words_and_tags[words] = tags

    return corpus_as_words_and_tags


if __name__ == '__main__':
    path_to_data = "./ud-treebanks-v2.4/UD_English-EWT/en_ewt-ud-train.conllu"
    corpus = load_unordered_data(path_to_data)
    print(random.choice(list(corpus.items())))





    # pretrained_encodings = load_encodings("wiki-news-300d-1M.vec")
    # print(pretrained_encodings["cat"])

