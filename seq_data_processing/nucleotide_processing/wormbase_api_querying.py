# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 12:33:02 2021

@author: Bill Sun
"""
# %%
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import pickle
# %%
# load all genes
dataset_path = "./tmp/clean_interventions_dataset.csv"

dataset = pd.read_csv(dataset_path)

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

unique_interventions = get_unique_interventions_and_count(dataset, 'Intervention(s)')

# checks for breaking for each variation
intervention_to_id = {}
failures = []
for variation in unique_interventions:
    base_url = 'https://wormbase.org/species/c_elegans/variation/'
    req = requests.get('%s%s' % (base_url, variation))
    if req.status_code != 200: # Not useful -> has error page
        print("%s not found" % (variation))
        break
    else:
        wormbaseVarId = req.url.split('/')[-1].split('?')[0]
        if wormbaseVarId == variation: # variation is not found in wormbase db
            failures.append(variation)
        else:
            intervention_to_id[variation] = wormbaseVarId
# %%
with open('./tmp/intervention_to_ids.pickle', 'wb') as handle:
    pickle.dump(intervention_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./tmp/interventions_without_ids.pickle', 'wb') as handle:
    pickle.dump(failures, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
with open('./tmp/intervention_to_ids.pickle', 'rb') as handle:
    intervention_to_id = pickle.load(handle)
# %%
id_to_intervention = {}
for iv, id in intervention_to_id.items():
    if id not in id_to_intervention:
        id_to_intervention[id] = iv
    else:
        print("??? Duplicate id: %s for %s, %s" % (id, iv, id_to_intervention[id]))
# %%
# returns the 

def wormbase_api_query(intervention_names):
    
    def wormbase_api_query_helper(wormbase_id):
        try:
            print('attempting to query: %s (%s)...' % (id_to_intervention[wormbase_id], wormbase_id))
            rtn = {}
            # get wild and mutant nucleotide seqs (REQUIRED)
            req_url = 'http://rest.wormbase.org/rest/field/variation/%s/sequence_context' % (wormbase_id)
            api_r = requests.get(req_url)
            output = api_r.json()
    
            rtn['wildtype'] = output["sequence_context"]["data"]["wildtype"]["positive_strand"]["sequence"]
            rtn['mutant'] = output["sequence_context"]["data"]["mutant"]["positive_strand"]["sequence"]
            # optional other stuff
            try:
                # get general type of variation (e.g. substitution, deletion) and gene it is located on
                req_url = 'http://rest.wormbase.org/rest/widget/variation/%s/overview' % (wormbase_id)
                api_r = requests.get(req_url)
                output = api_r.json()
                rtn['variation_type'] = output["fields"]["variation_type"]
                rtn['corresponding_gene'] = output["fields"]["corresponding_gene"]["data"]
            except (TypeError, KeyError):
                print("couldn't find overview field of %s (%s)" % (id_to_intervention[wormbase_id], wormbase_id))
                rtn['variation_type'] = {}
                rtn['corresponding_gene'] = {}
            try:
                # get location of intervention
                req_url = 'http://rest.wormbase.org/rest/field/variation/%s/genetic_position' % (wormbase_id)
                api_r = requests.get(req_url)
                output = api_r.json()
                rtn['genetic_position'] = output["genetic_position"]
            except (TypeError, KeyError):
                print("couldn't find genetic_position field of %s (%s)" % (id_to_intervention[wormbase_id], wormbase_id))
                rtn['genetic_position'] = {}
            return rtn
        
        except (TypeError, KeyError):
            print('ERROR! %s has some missing fields in the wormbase API' % (wormbase_id))
            return {}
    
    if type(intervention_names) == list:
        rtn, failures = {}, []
        for intervention_name in intervention_names:
            wormbase_id = intervention_to_id[intervention_name]
            output = wormbase_api_query_helper(wormbase_id)
            if output:
                rtn[intervention_name] = output
            else:
                failures.append((intervention_name, intervention_to_id[intervention_name]))
        print("-----\nThere were %d failures: %s" % (len(failures), str(failures)))
        
        return rtn
    
    elif type(intervention_names) == str:
        return wormbase_api_query_helper(intervention_to_id[intervention_names])

# %%
import random
test_ivs_10 = random.sample(list(intervention_to_id.keys()),10)
query_results = wormbase_api_query(test_ivs_10)
# %%
test_ivs_100 = random.sample(list(intervention_to_id.keys()),100)
query_results = wormbase_api_query(test_ivs_100)