# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 18:08:31 2021

@author: Bill Sun
"""

import pandas as pd
import numpy as np
import re
# %%
# cleans original sa dataset at dataset_path to output dataframe containing interventions vs percent change in lifespan 
# for all data instances; saved as csv in save_path, ignores specific interventions given by ignore array
# returns clean dataset
def clean_sa_dataset(dataset_path, save_path, ignore=['RNAi','OE']):
    raw_ds = pd.read_csv(dataset_path)
    # only include useful columns
    raw_ds = raw_ds[['Wild type lifespan (days)', 'Intervention on gene 1',
       'Gene 1 single mutant lifespan (days)', 'Intervention(s) on gene 2(,3)',
       'Gene(s) 2(,3) mutant lifespan (days)',
       'Double (triple) mutant (genes 1,2,(3)) lifespan (days)']]
    # must compare vs wildtype; so drop all rows w/o wildtype metric
    filtered_ds = raw_ds[raw_ds['Wild type lifespan (days)'].notna()]
    
    # convert dataset into suitable format, with each row being intervention(s) and corresponding wild/mutant lifespan
    subset_1 = filtered_ds[['Wild type lifespan (days)', 
                            'Intervention on gene 1',
                            'Gene 1 single mutant lifespan (days)']]
    subset_1 = subset_1.dropna()
    
    subset_2_3 = filtered_ds[['Wild type lifespan (days)',
                              'Intervention(s) on gene 2(,3)',
                              'Gene(s) 2(,3) mutant lifespan (days)']]
    subset_2_3 = subset_2_3.dropna()
    
    subset_1_2_3 = filtered_ds[['Wild type lifespan (days)',
                                'Intervention on gene 1',
                                'Intervention(s) on gene 2(,3)',
                                'Double (triple) mutant (genes 1,2,(3)) lifespan (days)']]
    subset_1_2_3 = subset_1_2_3.dropna()
    # create new column that contains all interventions (1,2,3)
    subset_1_2_3['All interventions'] = subset_1_2_3[['Intervention on gene 1','Intervention(s) on gene 2(,3)']].agg(';'.join, axis=1)
    # only include useful columns
    subset_1_2_3 = subset_1_2_3[['Wild type lifespan (days)',
                                 'All interventions',
                                 'Double (triple) mutant (genes 1,2,(3)) lifespan (days)']]

    # rename all columns for each subset
    for intervention_subset in [subset_1,subset_2_3,subset_1_2_3]:
        intervention_subset.columns=['Wild Lifespan', 'Intervention(s)', 'Mutant Lifespan']
    clean_ds=pd.concat([subset_1,subset_2_3,subset_1_2_3],axis=0, ignore_index=True)
    # swap columns
    clean_ds = clean_ds.reindex(columns=['Intervention(s)', 'Wild Lifespan', 'Mutant Lifespan'])
    
    # calculate percent change for each data instance
    clean_ds['Percent Change'] = clean_ds.apply(
        lambda row: round(100*(row['Mutant Lifespan'] - row['Wild Lifespan'])/row['Wild Lifespan'],3), axis=1
    )
    # drop dups
    # clean_ds = clean_ds.drop_duplicates().reset_index()[['Intervention(s)', 'Wild Lifespan', 'Mutant Lifespan','Percent Change']]
    
    # remove entries containing specific interventions
    for bad_intervention in ignore:
        clean_ds = clean_ds[clean_ds['Intervention(s)'].map(lambda x: bad_intervention not in str(x))]
    
    clean_ds = clean_ds.reset_index()[['Intervention(s)', 'Wild Lifespan', 'Mutant Lifespan','Percent Change']]
    
    # save entire dataset
    clean_ds.to_csv(save_path, index=False)
    
    return clean_ds

# %%
path = "../../data/SynergyAge_DS/SynergyAge_Database.csv"
#out_path = "../../data/SynergyAge_DS/interventions/sa_clean_interventions.csv"
out_path = "./tmp/clean_interventions_dataset.csv"
sa_clean_interventions = clean_sa_dataset(path, out_path)
# %%
# get some nice analytics
# value counts for entire set of interventions
sa_clean_interventions['Intervention(s)'].value_counts().reset_index().rename(
    columns={'index':'Intervention(s)', 'Intervention(s)':'Counts'}
    ).to_csv('../data_analytics/intervention_collective_value_counts.csv', index=False)
# %%
# value counts for individual intervention

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

intervention_individual_value_counts = get_unique_interventions_and_count(sa_clean_interventions, 'Intervention(s)')
pd.DataFrame.from_dict(intervention_individual_value_counts, orient="index").to_csv("../data_analytics/intervention_individual_value_counts.csv")