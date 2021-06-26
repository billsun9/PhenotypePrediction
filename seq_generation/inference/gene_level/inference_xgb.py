# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:05:57 2021

@author: Bill Sun
"""

import pickle
import pandas as pd
import numpy as np
import re
import sys

sys.path.append('./utils/')

from utils import genes_to_vec_np, rmse
# %%
save_path = "./models/xgb_opt_5_22_21.pickle"

with open(save_path, 'rb') as f:
    xgb_model = pickle.load(f)
# %% 
# params: genes knocked out (str or list); returns percent change in mutant vs wildtype(float)
# e.g. "sod-2;eat-2" --> 106.6
def predict(genes):
    if(type(genes) == list):
        rtn = []
        for gene_seq in genes:
            vec = genes_to_vec_np(gene_seq)
            rtn.append(xgb_model.predict(vec.reshape(1,vec.shape[0]))[0])
        return np.array(rtn)
    else:
        vec = genes_to_vec_np(genes)
        return xgb_model.predict(vec.reshape(1,vec.shape[0]))[0]
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
# %%
xgb_preds = xgb_model.predict(X)

print("RMSE %.4f" % (rmse(xgb_preds.reshape(xgb_preds.shape[0],), Y))) # 26.1261
# %%
# save the nn predictions as aditional column to csv
save_df = data_clean.copy(deep=True)
save_df['XGB Percent Change Predictions'] = xgb_preds
save_df.to_csv("./model_results/xgb_5_22_21_predictions.csv", index=False)