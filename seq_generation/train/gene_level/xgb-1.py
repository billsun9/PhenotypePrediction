# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:32:41 2021

@author: Bill Sun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
import sys
import sklearn
from xgboost import XGBClassifier, XGBRegressor

sys.path.append('../utils')

from utils import get_unique_genes, genes_to_vec_np, mae, rmse
# %%
clean_sa_ds_path = '../../data/SynergyAge_DS/sa_clean_data.csv'
data_clean = pd.read_csv(clean_sa_ds_path)
# %%
# creates the gene to int map for the cleaned dataframe
unique_genes = get_unique_genes(data_clean, 'Gene(s)')

save_path = "./synergyage_gene_to_int_map_REVISED.pickle"

with open(save_path, 'wb') as handle:
    pickle.dump({unique_genes[i]: i for i in range(len(unique_genes))}, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
X = np.array([genes_to_vec_np(gene_seq) for gene_seq in data_clean['Gene(s)']])
Y = data_clean.values[:,-1].astype(np.float32)
# %%
seed = 8
test_size=0.2
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# %%
xgb_model_1 = XGBRegressor(objective ='reg:linear', 
                           eta=0.1,
                           n_estimators=250,
                           subsample=1.0, 
                           seed=seed)
xgb_model_1.fit(X_train, Y_train)
# %%
pred = xgb_model_1.predict(X_test)

print("RMSE: %.3f" % (rmse(pred, Y_test)))
# %%
# import math
# print("RMSE: %.3f" % (math.sqrt(sklearn.metrics.mean_squared_error(Y_test, pred))))