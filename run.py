from collections import namedtuple
import sys
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import create_data as cd
from net import Net
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics

def train_epoch(X, Y, net, opt, criterion, batch_size=50):
    net.train()
    losses = []
    correct = 0
    truths = []
    predictions = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = net(x_batch)
        for i in range(len(y_hat)):
            truths.append(y_batch[i])
            if y_hat[i] >= 0.5:
                predictions.append(1)
                if y_batch[i] == 1:
                    correct += 1
            if y_hat[i] < 0.5:
                predictions.append(0)
                if y_batch[i] == 0:
                    correct += 1
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        losses.append(loss.data.numpy())
    accuracy = correct/len(Y)
    print("roc", metrics.roc_auc_score(np.array(truths), np.array(predictions)))
    print("accuracy", accuracy)
    return accuracy, losses

def test(X, Y, net):
    #print(X)
    #print(Y)
    correct = 0
    truths = []
    predictions = []
    for i in range(Y.size()[0]):
        truths.append(Y[i])
        y_hat = net.forward(X[i])
        if y_hat >= 0.5:
            predictions.append(1)
            if Y[i] == 1:
                correct += 1
        if y_hat < 0.5:
            predictions.append(0)
            if Y[i] == 0:
                correct += 1
    print("roc test", metrics.roc_auc_score(np.array(truths), np.array(predictions)))
    return correct/len(Y)

def main():
    print("Creating examples...")
    examples = cd.create_examples()
    np.random.shuffle(examples)
    print("Generating tensors...")
    X, Y = cd.generate_tensors(examples)
    #X = pickle.load(open('tf.p', 'rb'))
    #Y = pickle.load(open('tv.p', 'rb'))
    Y = Y.view(Y.size()[0], 1)
    #print(Y)

    net = Net(X.size()[1])
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    e_losses = []
    accuracy = []
    num_epochs = 5
    for e in range(num_epochs):
        a, losses = train_epoch(X[:9*X.size()[0]//10, :], Y[:9*Y.size()[0]//10, :], net, opt, criterion)
        e_losses += losses
        accuracy.append(a)
    print("accuracy test", test(X[9*X.size()[0]//10:, :], Y[9*Y.size()[0]//10:, :], net))
    plt.plot(accuracy)
    plt.show()
    #print(e_losses)b

if __name__ == '__main__':
    main()