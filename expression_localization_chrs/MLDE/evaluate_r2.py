# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 17:19:29 2023

@author: Bill Sun
"""
import numpy as np
import torch
from sklearn.metrics import r2_score

def r2_evaluate(model, dataloader):
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['sequence'].float()
            targets = batch['target'].view(-1, 1).float()
            outputs = model(inputs)

            all_predictions.append(outputs)
            all_targets.append(targets)

    # Concatenate predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    return r2_score(all_targets, all_predictions)