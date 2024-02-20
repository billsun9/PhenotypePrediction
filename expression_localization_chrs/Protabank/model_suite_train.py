# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:12:38 2023

@author: Bill Sun
"""
from Protabank.data_processing import getFilteredDataset
from Protabank.dataloaders import get_dataloaders, constructVocab
from Protabank.baselines.MLP import *
from Protabank.baselines.LSTM import *
from Protabank.train import train
from Protabank.utils import plotLosses

PATH = './Data/Protabank/Principles for computational design of binding antibodies.csv'
dataset = getFilteredDataset(PATH)
vocab = constructVocab(dataset) # should be 20 AA's + <sos> + <eos>
max_length = 300 # somewhat arbitrarily chosen
train_dataloader, test_dataloader = get_dataloaders(dataset, split_percent = 0.8, max_length = max_length)
# %%
input_size = max_length * len(vocab)

models = {"LR":LR(input_size = input_size, output_size = 1),
          "MLP_sm":MLP1(input_size = input_size, output_size = 1),
          "MLP_md":MLP2(input_size = input_size, output_size = 1),
          # "MLP_lg":MLP3(input_size = input_size, output_size = 1)
          }

lstm_models = {"LSTM1":LSTM1(input_size = input_size, output_size = 1),
               "BLSTM1":BLSTM1(input_size = input_size, output_size = 1),
          # "LSTM_BN1":LSTM_BN1(input_size = input_size, output_size = 1)
          }
# %%
for model_name, model in models.items():
    print("<" + "-"*25 + ">")
    train_losses, test_losses = train(model, train_dataloader, test_dataloader, verbose = True, num_epochs = 80)
    plotLosses(model_name, train_losses, test_losses)
    print("{} Loss: {}".format(model_name, test_losses[-1]))