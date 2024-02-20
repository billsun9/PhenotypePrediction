# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:12:36 2023

@author: Bill Sun
"""
import os
import pandas as pd
import numpy as np

def getRawDataset(path):
    print("Reading from: {}".format(path))
    return pd.read_csv(path)

def EDA1():
    from Protabank.utils import aa_map
    df = getRawDataset()
    
    AAS = set(aa_map.keys())
    
    for i in range(len(df)):
        ex = df.iloc[i]['Sequence']
        assert not set(ex).difference(AAS)
    
    print(list(df.columns))
    print("---"*5)
    print("Available Target ('Assay/Protocol') Variables:")
    for c in df['Assay/Protocol'].unique():
        print("{}: {}".format(c, len(df.loc[df['Assay/Protocol'] == c])))
    print("---"*5)
    dataset = df.loc[df['Assay/Protocol'] == 'Expression'].reset_index(drop=True)[['Sequence', 'Data']]
    
    lens = dataset['Sequence'].map(lambda x: len(x))
    print("Sequence Length statistics")
    print(lens.describe())
    print("---"*5)
    print("Expression Statistics")
    print(dataset['Data'].describe())
    print("---"*5)

# 
def getFilteredDataset(path):
    df = getRawDataset(path)
    return df.loc[df['Assay/Protocol'] == 'Expression'].reset_index(drop=True)[['Sequence', 'Data']]