# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:22:36 2021

@author: Bill Sun
"""

import pickle
import pandas as pd
import numpy as np
import re
import sys
import tensorflow as tf

sys.path.append('./utils/')

from utils import genes_to_vec_np, rmse, mae
# %%
save_path = "./models/nn_dense_5_24_21"

nn_model = tf.keras.models.load_model(save_path)
# %% 
# params: genes knocked out (str or list); returns percent change in mutant vs wildtype(float)
# e.g. "sod-2;eat-2" --> 106.6
def predict(genes):
    if(type(genes) == list):
        rtn = []
        for gene_seq in genes:
            vec = genes_to_vec_np(gene_seq)
            rtn.append(nn_model.predict(vec.reshape(1,vec.shape[0]))[0])
        return np.array(rtn)
    else:
        vec = genes_to_vec_np(genes)
        return nn_model.predict(vec.reshape(1,vec.shape[0]))[0]
# %%
print(predict("sod-2;eat-2"))
print(predict("pbs-2;glp-1"))
print(predict("pbs-3;glp-1"))
print(predict("glp-1"))
print(predict("nhr-80; glp-1"))
print(predict("nhr-80; glp-1; sod-2;eat-2; daf-16")) # Not in original dataset!
# %%
print(predict(["sod-2;eat-2","pbs-2;glp-1","pbs-3;glp-1","glp-1","nhr-80; glp-1"]))
# xgb_model can also directly be called on np array of gene data via xgb_model.predict(np.array)
# %%
# load entire sa dataset and predict on it
clean_sa_ds_path = '../data/SynergyAge_DS/sa_clean_data.csv'
data_clean = pd.read_csv(clean_sa_ds_path)

X = np.array([genes_to_vec_np(gene_seq) for gene_seq in data_clean['Gene(s)']])
Y = data_clean.values[:,-1].astype(np.float32)

nn_preds = nn_model.predict(X)

print("RMSE %.4f" % (rmse(nn_preds.reshape(nn_preds.shape[0],), Y))) #34.7411
# %%
# save the nn predictions as aditional column to csv
save_df = data_clean.copy(deep=True)
save_df['Percent Change Predictions'] = nn_preds.reshape(3885,)
save_df.to_csv("./model_results/nn_5_24_21_predictions.csv", index=False)