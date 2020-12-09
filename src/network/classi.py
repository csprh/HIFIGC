import torch
import torch.nn as nn
import torch.nn.functional as F
#from nn.flatten import Flatten
import numpy as np

class Classi(nn.Module):
    def __init__(self, image_dims, batch_size, activation='relu', C=101):

        super(Classi, self).__init__()
        #B, 220, 16,16 >  #B, 8, 16,16

        self.b1 = nn.BatchNorm1d(8)
        self.d1 = nn.Dropout(0.7)
        self.lr1 = nn.LeakyReLU(0.1)
        self.l1 = nn.Linear(56320,8) #8*12*12
        self.l2 = nn.Linear(8,C) #8*12*12


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = self.lr1(x)
        x = self.b1(x)
        x = self.d1(x)


        x = self.l2(x)
        #x = F.softmax(x)
        return x

class ClassiOld(nn.Module):
    def __init__(self, image_dims, batch_size, activation='relu', C=101):

        super(Classi, self).__init__()
        #B, 220, 16,16 >  #B, 8, 16,16
        self.c1 = nn.Conv2d(220, 8, kernel_size=(5,5), stride=1)
        self.b1 = nn.BatchNorm2d(8)
        self.d1 = nn.Dropout2d(0.5)
        self.l1 = nn.Linear(1152,C) #8*12*12

    def forward(self, x):
        x = F.relu(self.c1(x))
        #x = self.b1(x)
        x = self.d1(x)
        #x = F.dropout(x, training=self.training)
        x = x.view(x.shape[0], -1)

        x = self.l1(x)
        result = F.softmax(x)
        return x


if __name__ == "__main__":
    B = 2
    C = 101
    print('Test Batch 1')
    x = torch.randn((B,220,16,16))
    x_dims = tuple(x.size())
    C = Classi(batch_size=B, C=101)
    C(x)

