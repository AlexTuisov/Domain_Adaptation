import torch
import torch.nn as nn
import English_only.Preprocessing as pr
import random
from gensim.models.wrappers import FastText

PATH_TO_DATA = "train.conllu"
PATH_TO_MODEL = "cc.en.300.bin"



def create_tensor(row_of_words, row_of_tags, model):
    list_of_tensors = []
    for word in row_of_words:
        try:
            vector = model[word]
        except KeyError:
            print("have a key error there!")
            print(word)
        tensor = torch.from_numpy(vector).float()
        list_of_tensors.append(tensor)
    output_tensor = torch.cat(list_of_tensors, 0)
    print(output_tensor)
    output_tensor = output_tensor.view(len(list_of_tensors), 1, len(list_of_tensors[0]))

    return output_tensor



# def train_model(train_dataset, word_embedding):
#     model = GRUPredictor(WORDS_EMBEDDING_DIMENSION, len(train_dataset.tag_set), 8)
#     model.to(DEVICE)
#     criterion = nn.NLLLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     all_losses = []
#     for epoch in range(1, NUMBER_OF_EPOCHS + 1):
#         average_loss = train_one_epoch(epoch, train_dataset, optimizer, model, criterion, word_embedding)
#         all_losses.append(average_loss)
#         print(f"This epoch the average loss was {average_loss:.4f}")
#
#     return model
#
#
# def train_one_epoch(epoch_number, train_dataset, optimizer, model, criterion, word_embedding):
#     random_order = list(range(len(train_dataset)))
#     random.shuffle(random_order)
#     epoch_loss = 0
#     row_counter = 0
#     for i in random_order:
#         batch = [word_embedding[train_dataset[i][j]] for j in range(len(train_dataset[i]))]
#         true_answers = [train_dataset[i][0][1]] * len(batch)
#         true_scores = map(calculate_true_score, zip(batch, true_answers))
#         output, loss = train_batch(optimizer, model, criterion, batch, true_scores)
#         epoch_loss += loss
#
#     average_loss = epoch_loss / row_counter
#     print(f"This epoch the average loss was {average_loss:.4f}")
#     return average_loss
#
#
# def train_batch(optimizer, model, criterion, batch, true_scores):
#     optimizer.zero_grad()
#     hidden = model.initHidden(batch.size()[0]).to(DEVICE)
#     output = model(batch, hidden)
#     loss = criterion(output, true_scores)
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
#     optimizer.step()
#     return output, loss.item()
#
#
# def calculate_true_score(sentence_and_tags, true_tags):
#     tags = sentence_and_tags[1]
#     length = len(tags)
#     assert length == len(true_tags)
#     count = 0
#     for i in range(length):
#         if tags[i] == true_tags[i]:
#             count += 1
#     return count/length


if __name__ == '__main__':
    language_model = FastText.load_fasttext_format(PATH_TO_MODEL)
    x, y, tag_set = pr.load_train_test_validation_sets(PATH_TO_DATA)
    train_set = pr.WordsDataset(x, tag_set)
    test_set = pr.WordsDataset(y, tag_set)

    item = train_set[16][0]["words"]
    print(item)
    my_little_tensor = create_tensor(item, None, language_model)
    print(my_little_tensor.shape)
    print("-----------------")

    model = train_model(train_set, language_model)

