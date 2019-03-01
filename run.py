from collections import namedtuple
import sys
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import optim
import numpy as np
import create_data as cd
from net import Net
from torch.autograd import Variable
#import matplotlib.pyplot as plt

def train_epoch(X, Y, net, opt, criterion, batch_size=50):
    net.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = net(x_batch)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        losses.append(loss.data.numpy())
    return losses

def main():
    print("Creating examples...")
    examples = cd.create_examples()
    np.random.shuffle(examples)
    print("Generating tensors...")
    X, Y = cd.generate_tensors(examples)
    Y = Y.view(Y.size()[0], 1)

    net = Net(X.size()[1])
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    e_losses = []
    num_epochs = 20
    for e in range(num_epochs):
        e_losses += train_epoch(X, Y, net, opt, criterion)
    #plt.plot(e_losses)
    print(e_losses)



if __name__ == '__main__':
    main()
