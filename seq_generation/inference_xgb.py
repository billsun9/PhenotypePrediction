# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:05:57 2021

@author: Bill Sun
"""

import pickle
import pandas as pd
import numpy as np
import re

save_path = "./models/xgb_opt_5_18_21.pickle"

with open(save_path, 'rb') as f:
    xgb_model = pickle.load(f)

def genes_to_vec_np(input_genes):
    with open('./train/synergyage_gene_to_int_map_REVISED.pickle', 'rb') as handle:
        alphabet = pickle.load(handle)
    cur_genes = input_genes.replace(" ","")
    genes = re.split(";|,",cur_genes)
    try:
        indices = [alphabet[gene] for gene in genes]
        vec = np.array([1 if i not in indices else 0 for i in range(len(alphabet.keys()))],
                          dtype='int32')
        return vec
    except KeyError:
        raise Exception("Invalid Input: ", genes)
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
print(predict("glp-1"))
print(predict("nhr-80; glp-1"))
print(predict("nhr-80; glp-1; sod-2;eat-2; daf-16"))
# %%
print(predict(["sod-2;eat-2","glp-1","nhr-80; glp-1"]))
# xgb_model can also directly be called on np array of gene data via xgb_model.predict(np.array)