from collections import namedtuple
import sys
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import create_data as cd
#import matplotlib.pyplot as plt

examples = cd.create_examples()
np.random.shuffle(examples)
#train_examples = examples[:9*len(examples)//10]
#test_examples = examples[9*len(examples)//10:]
X, Y = cd.generate_tensors(examples)
Y = Y.view(Y.size()[0], 1)
print(X.size())
print(Y.size())

class Net(nn.Module):
    def __init__(self, vocab_size, dropout=0.2):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(in_features=vocab_size, out_features=50)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_activation = nn.Sigmoid()

    def forward(self, input) -> torch.Tensor:
        a1 = self.linear1(input)
        h1 = self.relu1(a1)
        h1_dropout = self.dropout(h1)
        a2 = self.linear2(h1_dropout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_activation(a3)
        return y

net = Net(X.size()[1])
opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.BCELoss()

def train_epoch(model, opt, criterion, batch_size=X.size()[0]):
    model.train()
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

e_losses = []
num_epochs = 20
for e in range(num_epochs):
    e_losses += train_epoch(net, opt, criterion)
#plt.plot(e_losses)
print(e_losses)
