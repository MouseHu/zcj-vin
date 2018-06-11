import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
#from visualize import make_dot

class VINdyn(nn.Module):
    def __init__(self, config):
        super(VINdyn, self).__init__()
        self.config = config
        self.h = nn.Conv2d(
            in_channels=config.l_i,
            out_channels=config.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.t=nn.Conv2d(
            in_channels=config.l_h,
            out_channels=config.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=config.l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=config.l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.fc = nn.Linear(in_features=config.l_q, out_features=8, bias=False)
        self.w = Parameter(
            torch.randn(config.l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, X, S1,S2, config):
        h = self.h(X)
        t = h
        r = self.r(t)
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, config.k - 1):
            
            q = F.conv2d(
                torch.cat([r, v], 1),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)
            #print(torch.cat([r, v], 1).shape)
            #print(torch.cat([self.q.weight, self.w], 1).shape)
            v, _ = torch.max(q, dim=1, keepdim=True)
            t=self.t(t)
            r=self.r(t)

        q = F.conv2d(
            torch.cat([r, v], 1),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
            padding=1)
        #print(q.shape)

        slice_s1 = S1.long().expand(config.imsize, 1, config.l_q, q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        #print(slice_s1.shape)
        q_out = q.gather(2, slice_s1).squeeze(2)
        #print(q_out.shape)
        slice_s2 = S2.long().expand(1, config.l_q, q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q_out = q_out.gather(2, slice_s2).squeeze(2)
        #print(q_out.shape)
        logits = self.fc(q_out)
        #print(logits.shape)
        return logits, self.sm(logits)
class Config(object):
    def __init__(self):
        self.l_i=2
        self.l_h=150
        self.l_q=10
        self.batch_size=32
        self.k=3
        self.imsize=8
        self.lr=5e-4