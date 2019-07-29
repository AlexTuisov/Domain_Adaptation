import io
import glob
import os
import conllu
from pathlib import Path


def load_encodings(file_name):
    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

# def parse_conllu():
#     # some_file.py
#     import sys
#     # insert at 1, 0 is the script path (or '' in REPL)
#     sys.path.insert(1, '/path/to/application/app/folder')
#
#     import file


if __name__ == '__main__':
    data_file = open("./ud-treebanks-v2.4/UD_English-ESL/en_esl-ud-train.conllu", "r", encoding="utf-8")
    training_set = list(conllu.parse_tree_incr(data_file))
    my_token = training_set[16]
    sentence_as_dict = my_token.token
    print(sentence_as_dict)


    # pretrained_encodings = load_encodings("wiki-news-300d-1M.vec")
    # print(pretrained_encodings["cat"])

