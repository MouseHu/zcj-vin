import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
#from visualize import make_dot

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.conv_1 = nn.Conv2d(
            in_channels=config.l_i,
            out_channels=config.l1,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, ceil_mode=False)
        self.conv_2 = nn.Conv2d(
            in_channels=config.l1,
            out_channels=config.l1,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.pool_2=nn.MaxPool2d(kernel_size=1, ceil_mode=False)
        self.conv_3 = nn.Conv2d(
            in_channels=config.l1,
            out_channels=config.l2,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.pool_3=nn.MaxPool2d(kernel_size=2, ceil_mode=False)
        self.conv_4 = nn.Conv2d(
            in_channels=config.l2,
            out_channels=config.l2,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.pool_4=nn.MaxPool2d(kernel_size=1, ceil_mode=False)
        self.conv_5 = nn.Conv2d(
            in_channels=config.l2,
            out_channels=config.l2,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.pool_5=nn.MaxPool2d(kernel_size=1, ceil_mode=False)
        self.fc = nn.Linear(in_features=config.l2*(config.imsize/4)**2, out_features=8, bias=False)
       
        self.sm = nn.Softmax(dim=1)

    def forward(self, X, S1,S2, config):
        place=torch.zeros(X.shape[0],1,self.config.imsize,self.config.imsize)
        for i in range(S1.shape[0]):
            place[i,0,S1[i].long(),S2[i].long()]=1
        
        inputs=torch.cat((X,place),1)
        conv1=self.conv_1(inputs)
        pool1=self.pool1(conv1)
        conv2=self.conv_2(pool1)
        pool2=self.pool_2(conv2)
        conv3=self.conv_3(pool2)
        pool3=self.pool_3(conv3)
        conv4=self.conv_4(pool3)
        pool4=self.pool_4(conv4)
        conv5=self.conv_5(pool4)
        pool5=self.pool_5(conv5)
        out=self.fc(pool5.reshape(X.shape[0],pool5.shape[1]*pool5.shape[2]*pool5.shape[3]))
        return out, self.sm(out)
class Config(object):
    def __init__(self):
        self.l_i=3
        self.l2=100
        self.l1=50
        self.l_q=10
        self.batch_size=32
        self.k=3
        self.imsize=8
        self.lr=5e-4