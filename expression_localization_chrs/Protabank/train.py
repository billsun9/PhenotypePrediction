# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 20:07:58 2023

@author: Bill Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

def train(model,
          train_dataloader,
          test_dataloader,
          verbose = False,
          num_epochs = 100):
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(),lr=0.001)
    print("Model: ", model)
    if verbose:
        print("Num Epochs: ", num_epochs)
        print("Loss FN:",criterion)
        print("Optimizer:",optimizer)
    # Training loop
    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
    
        for batch in train_dataloader:
            inputs = batch['sequence'].float()
            targets = batch['target'].view(-1, 1).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        average_loss = total_loss / len(train_dataloader)
        if epoch % 10 == 0 and verbose:
            print(f'Train Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
        train_losses.append(average_loss)
        
        # Evaluation phase
        model.eval()
        total_eval_loss = 0
    
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch['sequence'].float()
                targets = batch['target'].view(-1, 1).float()
                outputs = model(inputs)
                eval_loss = criterion(outputs, targets)
                total_eval_loss += eval_loss.item()
    
        average_eval_loss = total_eval_loss / len(test_dataloader)
        if epoch % 10 == 0 and verbose:
            print(f'Validation Epoch [{epoch + 1}/{num_epochs}], Loss: {average_eval_loss:.4f}')
        test_losses.append(average_eval_loss)
        
    return train_losses, test_losses