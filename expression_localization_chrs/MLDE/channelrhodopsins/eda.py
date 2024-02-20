# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:08:51 2023

@author: Bill Sun
"""

import pandas as pd
import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def check(df):
    badIdxs = set()
    for i in range(len(df)):
        ex = df.iloc[i]
        if len(set([s.lower() for s in ex['sequence']])) != 4:
            print("Sussy behavior", i, set(ex['sequence']))
            break
        
        if not is_number(ex['mKate_mean']):
            # print(i, ex['mKate_mean'])
            badIdxs.add(i)
        if not is_number(ex['GFP_mean']):
            # print(i, ex['GFP_mean'])
            badIdxs.add(i)
        if not is_number(ex['intensity_ratio_mean']):
            # print(i, ex['intensity_ratio_mean'])
            badIdxs.add(i)
    return badIdxs

def clean(df, target):
    print("Cleaning Dataset...")
    print("Target Variable: {}".format(target))
    assert target in df.columns
    badIdxs = check(df)
    df = df.drop(labels=badIdxs)
    df['sequence'] = df['sequence'].apply(lambda x: x.upper())
    df = df.reset_index(drop=True)
    df = df.rename(columns={"sequence":"Sequence", target: "Data"}, errors="raise")
    df['Sequence'] = df['Sequence'].astype(str)
    df['Data'] = df['Data'].astype(float)
    df = df[['Sequence','Data']]
    print("Dataframe columns: {}".format(df.columns))
    print("No Samples: {}".format(len(df)))
    return df
# %%
d = {}

for num in q:
    try: d[num] += 1
    except KeyError: d[num] = 1


