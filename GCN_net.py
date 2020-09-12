from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import GConv

class GCN7(nn.Module):
    def __init__(self, channel=4, lych = 10):
        super(GCN7, self).__init__()     
        self.channel = channel
        self.basic_layers = lych 
  
        self.model = nn.Sequential(
            GConv(1, lych, 5, padding=2, stride=1, M=channel, nScale=1, bias=False, expand=True),
            nn.BatchNorm2d(lych*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),              # output = 45*45

            GConv(lych, 2*lych, 5, padding=2, stride=1, M=channel, nScale=2, bias=False),
            nn.BatchNorm2d(2*lych*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),           # output = 22*22

            GConv(2*lych, 4*lych, 5, padding=2, stride=1, M=channel, nScale=3, bias=False),
            nn.BatchNorm2d(4*lych*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),           # output = 11*11

            GConv(4*lych, 8*lych, 5, padding=2, stride=1, M=channel, nScale=4, bias=False),
            nn.BatchNorm2d(8*lych*channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),          # output = 5*5
        )
        self.fc1 = nn.Linear(8*lych*5*5, 2000, bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2_7 = nn.Linear(2000, 7, bias=False)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 8*self.basic_layers, self.channel, 5, 5)   
        x = torch.max(x, dim=2)[0]        
        x = x.view(x.shape[0],-1)        
        x = self.fc1(x)        
        x = self.relu(x)        
        x = self.dropout(x)         
        x = self.fc2_7(x)
        return x
