import numpy as np
import sys
import csv
#import matplotlib.pyplot as plt
import math
import re
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pickle

csv.field_size_limit(sys.maxsize)
input_file = 'data/all_data_v3.csv'
DOCUMENT_IND = 2
SECTIONS_START_IND = 8
OUTCOME_IND = 4
f = open("data/lsa_popular_words_all.txt", "r")
vocab = {}
vocab_ind2word = {}
index = 0
for line in f:
    vocab[line.strip()] = index
    vocab_ind2word[index] = line.strip()
    index += 1
regex = re.compile('[^a-zA-Z]')
stopwords = stopwords.words('english')


'''
    1 for mentioned, 0 for not mentioned, for each section
'''
def section_features(row):
    features = []
    for i in row[SECTIONS_START_IND:]:
        if i == '':
            features.append(0)
        else:
            features.append(float(i))
    return np.array(features)

'''
    Features: word counts, sections, and clustering info 
'''
def feature_extractor(row):
    features = np.zeros((len(vocab)))

    f1 = section_features(row)

    for word in row[DOCUMENT_IND].split():
        word = regex.sub('', word.lower())
        if (word not in stopwords) and (word != '') and (word in vocab):
            features[vocab[word]] += 1
    return features

def generate_tensors(train_examples):
    tensors_features = []
    tensors_values = []
    print("Extracting features...")
    for row,value in train_examples:
        features = feature_extractor(row)
        tensors_features.append(features)
        tensors_values.append(value)
    tensors_features = torch.Tensor(tensors_features)
    tensors_values = torch.Tensor(tensors_values)
    #pickle.dump(tensors_features, open('tf.p', 'wb')) 
    #pickle.dump(tensors_values, open('tv.p', 'wb')) 
    return tensors_features, tensors_values

#######################################################################################

def create_examples():
    examples = []
    with open(input_file, encoding='ISO-8859-1') as input:
        reader = csv.reader(input)
        next(reader)
        n = 0
        for row in reader:
            #if matrix[n] == 0:
            value = 0
            if "APPROVED" in row[OUTCOME_IND] or "CONCURRED" in row[OUTCOME_IND]:
                value = 1
            examples.append((row, value))   
            n += 1 
    return np.array(examples)

"""
def main(args):
    examples = create_examples()
    np.random.shuffle(examples)
    # train_examples = examples[:7*num_rows//10]
    # test_examples = examples[7*num_rows//10:]
    train_examples = examples[:9*len(examples)//10]
    test_examples = examples[9*len(examples)//10:]
    generate_tensors(train_examples)
"""