# -*- coding: utf-8 -*-
"""
Created on Tue May 18 18:26:21 2021

@author: Bill Sun
"""
import pickle
import re
import numpy as np
def vec_to_genes(input_vec): # e.g. "sod-2;eat-2" --> np.array([1,1,1,1,0,1,...], dtype=int32)
    with open('./train/synergyage_gene_to_int_map_REVISED.pickle', 'rb') as handle:
        alphabet = pickle.load(handle)
    locs = np.where(input_vec == 0)
    rtn = []
    for loc in locs:
        for gene, idx in alphabet.items():
            if loc == idx:
                rtn.append(gene)
                
    return rtn
# %%
# give to vic; 5/18/21
pred2 = xgb_model.predict(X)
# %%
import pandas as pd

data_clean2 = data_clean.copy(deep=True)
data_clean2 = data_clean2.reset_index() # gets rid of weird indexing problem; should have done earlier
# %%
data_clean2['Preds'] = pd.Series(pred2)
# %%

# %%
data_clean3 = data_clean2[['Gene(s)', 'Wild Lifespan', 'Mutant Lifespan','Percent Change', 'Preds']]
data_clean3.to_csv("SA_mutant_percent_lifespan_change_preds2.csv", index=False)