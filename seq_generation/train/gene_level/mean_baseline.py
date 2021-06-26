# -*- coding: utf-8 -*-
"""
Created on Thu May 20 21:41:42 2021

@author: Bill Sun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle
import sys

sys.path.append('../utils')

from utils import get_unique_genes, genes_to_vec_np, mae, rmse
# %%
data_path = '../../data/SynergyAge_DS/sa_clean_data.csv'
data = pd.read_csv(data_path)

X = np.array([genes_to_vec_np(gene_seq) for gene_seq in data['Gene(s)']])
Y = data.values[:,-1].astype(np.float32)
# %%
import sklearn
from sklearn.model_selection import train_test_split
seed = 88
test_size=0.15
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# %%
# mean
y_mean = np.mean(Y)
Y_pred = np.array([y_mean for _ in range(len(Y_test))])

print("RMSE %.3f" % (rmse(Y_pred, Y_test)))  # 58.396