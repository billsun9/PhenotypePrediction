# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 11:43:40 2021

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

base_url = 'https://wormbase.org/species/c_elegans/variation/'

variation = 'e1370'

url = base_url+variation
req = requests.get(url)

print(req.history) # if no redirect: [], if redirect: status code [307]
print(req.url) # new url??

tmp = req.url
wormbaseVarId = tmp.split('/')[-1].split('?')[0]

api_r = requests.get('http://rest.wormbase.org/rest/field/variation/%s/sequence_context' % (wormbaseVarId))
# %%
with open('./tmp/intervention_to_ids.pickle', 'rb') as handle:
    intervention_to_id = pickle.load(handle)
    
# %%
test_ivs = ["ad609", "ad1116", "ak47", "cxTi9279"]
test_ivs_ids = [intervention_to_id[iv] for iv in test_ivs]

test_id = test_ivs_ids[3]
req_url = 'http://rest.wormbase.org/rest/field/variation/%s/sequence_context' % (test_id)
api_r = requests.get(req_url)
output = api_r.json()
# %%
wild = output['sequence_context']['data']['wildtype']['positive_strand']['sequence']
mutant= output['sequence_context']['data']['mutant']['positive_strand']['sequence']
# %%

t1 = set(list(key.lower() for key in intervention_to_id.keys()))