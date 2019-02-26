import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import math
import re
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

csv.field_size_limit(sys.maxsize)
input_file = 'data/all_data_v3.csv'
DOCUMENT_IND = 2
OUTCOME_IND = 4
f = open("lsa_words/lsa_popular_words_all.txt", "r")
vocab = {}
vocab_ind2word = {}
index = 0
for line in f:
    vocab[line.strip()] = index
    vocab_ind2word[index] = line.strip()
    index += 1
regex = re.compile('[^a-zA-Z]')
stopwords = stopwords.words('english')

# feature is list of sections, 1 for mentioned, 0 for not mentioned
def feature_extractor(row):
    features = np.zeros((len(vocab)))
    # creates a feature array using the sections + clustering info
    for word in row[DOCUMENT_IND].split():
        word = regex.sub('', word.lower())
        if (word not in stopwords) and (word != '') and (word in vocab):
            features[vocab[word]] += 1
    return features

def generate_train(train_examples):
    for row,value in train_examples:
        x += 1
        features = feature_extractor(row)
        features = torch.Tensor(features)
        print(features)

#######################################################################################

def main(args):
    num_rows = row_count(input_file)
    examples = create_examples()
    np.random.shuffle(examples)
    # train_examples = examples[:7*num_rows//10]
    # test_examples = examples[7*num_rows//10:]
    train_examples = examples[:9*len(examples)//10]
    test_examples = examples[9*len(examples)//10:]
    generate_train(train_examples)
    

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)