import torch
import torch.nn as nn
import torch.nn.functional as F
#from nn.flatten import Flatten
import numpy as np


class Classi(nn.Module):
    def __init__(self, image_dims, batch_size, activation='relu', C=101,
                 channel_norm=True):

        super(Classi, self).__init__()
        #B, 220, 16,16 >  #B, 8, 16,16
        self.c1 = nn.Conv2d(220, 8, kernel_size=(5,5), stride=1)
        self.b1 = nn.BatchNorm2d(8)
        self.d1 = nn.Dropout2d(0.5)
        #self.f1 = nn.Flatten(2048)
        self.l1 = nn.Linear(2048,C)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.b1(x)
        x = F.dropout(x, training=self.training)
        x = x.view(x.shape[0], -1)
        #x = x.view(-1, self.num_flat_features(x))
        x = self.l1(x)
        result = F.softmax(x)
        return result

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    B = 2
    C = 101
    print('Test Batch 1')
    x = torch.randn((B,220,16,16))
    x_dims = tuple(x.size())
    C = Classi(image_dims=x_dims[1:], batch_size=B, C=101)
    C(x)

