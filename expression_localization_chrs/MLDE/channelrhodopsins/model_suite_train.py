# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:12:38 2023

@author: Bill Sun
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from MLDE.channelrhodopsins.dataloaders import get_dataloaders, constructVocab
from MLDE.models.MLP import *
from MLDE.models.LSTM import *
from MLDE.models.CNN import *
from MLDE.channelrhodopsins.train import train
from Protabank.utils import plotLosses

from MLDE.channelrhodopsins.data_preprocessing import clean
from MLDE.evaluate_r2 import r2_evaluate
PATH = './Data/ChR/pnas.1700269114.sd01.csv'

dataset = pd.read_csv(PATH)
dataset = dataset[['sequence','mKate_mean','GFP_mean', 'intensity_ratio_mean']]
dataset = clean(dataset, 'mKate_mean')
# tee hee
dataset['Data'] = 100 * dataset['Data']
# %%
BATCH_SIZE = 8
vocab = constructVocab(dataset) # should be 4 Nucleotides's + <sos> + <eos>
# print(vocab.get_itos())
# df['Sequence'].apply(lambda x: len(str(x))).describe()
max_length = 1100 # somewhat arbitrarily chosen
train_dataloader, test_dataloader = get_dataloaders(dataset, format="cgr", batch_size=BATCH_SIZE, split_percent = 0.8, max_length = max_length)
# %%
input_size = max_length * len(vocab)

linear_models = {
    "LR":LR(input_size = input_size, output_size = 1),
    "MLP_sm":MLP1(input_size = input_size, output_size = 1),
    "MLP_md":MLP2(input_size = input_size, output_size = 1),
    "MLP_lg":MLP3(input_size = input_size, output_size = 1)
    }

lstm_models = {
    "LSTM1":LSTM1(input_size = input_size, output_size = 1),
    "LSTM2":LSTM2(input_size = input_size, output_size = 1),
    "BLSTM1":BLSTM1(input_size = input_size, output_size = 1),
          # "LSTM_BN1":LSTM_BN1(input_size = input_size, output_size = 1)
          }

cnn_models = {
    "CNN1": EvenShallowerCNN(),
    "CNN2": ShallowCNN(),
    "CNN3": DeepCNN()
    }

loss_data = {
    # Add more experiments as needed
}

# %%
def model_eval(model_dict):
    for name in model_dict:
        output = r2_evaluate(model_dict[name], train_dataloader)
        print("Train {} R^2 Score:".format(name), round(output,6))
    for name in model_dict:
        output = r2_evaluate(model_dict[name], test_dataloader)
        print("Test {} R^2 Score:".format(name), round(output,6))
# %%
# DATALOADERS MUST BE SET TO PROPER FORMAT!!!!!!!!!!

def train_eval(model_dict, num_epochs = 50, retry=False):
    # Your dictionary mapping experiment to train and val losses
    # experiments = [model_name + "_e=" + str(num_epochs) for model_name in model_dict]
    for model_name, model in model_dict.items():
        print("<" + "-"*25 + ">")
        if model_name + "_e=" + str(num_epochs) in loss_data and not retry: continue
        train_losses, test_losses = train(model, train_dataloader, test_dataloader, verbose = True, num_epochs = num_epochs)
        loss_data[model_name + "_e=" + str(num_epochs)] = {'train': train_losses, 'val': test_losses}
        print("[{} Val Loss: {}]".format(model_name, test_losses[-1]))
    model_eval(model_dict)

def plot_graphs(model_dict, num_epochs):
    # Iterate through experiments and plot on subplots
    datapoints = {model_name+"_e="+str(num_epochs):loss_data[model_name+"_e="+str(num_epochs)] for model_name in model_dict}
    # print(datapoints)
    print(len(datapoints))
    # Create subplots with a single row
    num_experiments = len(model_dict)
    fig, axes = plt.subplots(ncols=num_experiments, figsize=(4 * num_experiments, 4))
    
    for i, (experiment, losses) in enumerate(datapoints.items()):
        # Plot training loss
        axes[i].plot(losses['train'], label='Train Loss', marker='o')
        
        # Plot validation loss
        axes[i].plot(losses['val'], label='Validation Loss', marker='o')
        
        # Set subplot title and labels
        axes[i].set_title(experiment)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        
        # Display legend
        axes[i].legend()
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    # plt.suptitle("MLPs Performance in Predicting Expression")
    # Show the plot
    plt.show()