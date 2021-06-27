# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:32:59 2021

@author: Bill Sun
"""
import numpy as np
import re
import pandas as pd
import os
import sys
import pickle

# takes 2 numpy vectors; calculates rmse
def rmse(y_hat, y):
    if y_hat.shape != y.shape:
        raise Exception("Shapes of input vectors dont match: {}, {}".format(y_hat.shape, y.shape))
    if y_hat.ndim != y.ndim:
        raise Exception("Dimensions of input vectors dont match: {}, {}".format(y_hat.ndim, y.ndim))
    y_hat, y = y_hat.astype('float32'), y.astype('float32')
    return np.sqrt((np.square(y - y_hat)).mean())

# takes 2 numpy vectors; calculates mae
def mae(y_hat, y):
    if y_hat.shape != y.shape:
        raise Exception("Shapes of input vectors dont match: {}, {}".format(y_hat.shape, y.shape))
    if y_hat.ndim != y.ndim:
        raise Exception("Dimensions of input vectors dont match: {}, {}".format(y_hat.ndim, y.ndim))
    y_hat, y = y_hat.astype('float32'), y.astype('float32')
    return (np.absolute(y - y_hat)).mean()

# remove weird named genes (capitals), calculates percent chagne, removes duplicates
def clean_data_and_add_output(df): 
    data_c = df.copy(deep=True)
    data_clean = data_c[~data_c['Gene(s)'].str[:].str.contains(".", regex=False)]
    data_clean['Percent Change'] = data_clean.apply(
        lambda row: round(100*(row['Mutant Lifespan'] - row['Wild Lifespan'])/row['Wild Lifespan'],3), 
        axis=1
        )
    
    return data_clean.drop_duplicates().reset_index()[['Gene(s)', 'Wild Lifespan', 'Mutant Lifespan','Percent Change']]

# takes dataframe and column name (probably 'Gene(s)') and return list of all unique genes
def get_unique_genes(df, col_name):
    l = []
    for cur_genes in list(df[str(col_name)]):
        cur_genes = cur_genes.replace(" ","")
        genes = re.split(";|,",cur_genes)
        for gene in genes:
            if gene not in l:
                l.append(gene)
    return l

# e.g. "sod-2;eat-2" --> np.array([1,1,1,1,0,1,...], dtype=int32)
def genes_to_vec_np(input_genes): 
    with open('./synergyage_gene_to_int_map_REVISED.pickle', 'rb') as handle:
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

# e.g. "np.array([1,1,1,1,0,1,...], dtype=int32) --> "sod-2;eat-2"
def vec_to_genes_np(input_vec): 
    with open('./synergyage_gene_to_int_map_REVISED.pickle', 'rb') as handle:
        alphabet = pickle.load(handle)
    locs = np.where(input_vec == 0)[0]
    rtn = []
    for loc in locs:
        for gene, idx in alphabet.items():
            if loc == idx:
                rtn.append(gene)
                
    return ";".join(rtn)

# %% interventions
# 6/25/21
# iterates through df[col_name] and return frequency of each intervention as dict
def get_unique_interventions_and_count(df, col_name):
    d = {}
    for cur_interventions in list(df[str(col_name)]):
        cur_interventions = cur_interventions.replace(u'\xa0', ' ') # gets rid of '\xa0' non-breaking space issue
        cur_interventions = cur_interventions.replace(" ","")
        interventions = re.split(";|,",cur_interventions)
        for intervention in interventions:
            if intervention not in d:
                d[intervention] = 1
            else:
                d[intervention] += 1
    return d
