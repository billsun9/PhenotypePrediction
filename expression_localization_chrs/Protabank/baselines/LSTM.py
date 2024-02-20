# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 22:13:07 2023

@author: Bill Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM1(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM1, self).__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        # Take the output from the last time step
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class BLSTM1(nn.Module):
    def __init__(self, input_size, output_size):
        super(BLSTM1, self).__init__()
        self.lstm = nn.LSTM(input_size, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * 256, 128), # BIDIRECTIONAL
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        # Global average pooling
        x = torch.cat((x[:, -1, :256], x[:, 0, 256:]), dim=1)
        x = self.fc(x)
        return x

# DOESNT WORK
class LSTM_BN1(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM_BN1, self).__init__()
        self.lstm = nn.LSTM(input_size, 512, batch_first=True)
        self.bn = nn.BatchNorm1d(512)
        
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.bn(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x