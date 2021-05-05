# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 22:41:24 2021

@author: Bill Sun
"""

ex_file_path = '../data/Example_Animal_Genomes/anolis carolinensis (7.2 years max)/chrMT.fna'

f = open(ex_file_path, 'r')

nucleotide_seq_mt = ""
numLines = 0
for line in f:
    nucleotide_seq_mt+=line.rstrip('\n')
    numLines += 1
f.close()

print("there are {} lines in the mt chromosome, with {} nucleotides".format(numLines, len(nucleotide_seq_mt)))
# %%
# one hot encoding via sklearn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# DONT USE THIS
# takes seq as str input, return (len(str) x # categories) np.array
def onehot_encode_seq(input_seq):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)
    
    integer_encoded = label_encoder.fit_transform(np.array(list(input_seq)))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
# %%
seq1 = 'AACG'
seq2 = 'AGC'
seq3 = 'TTT'
encoded_data1 = onehot_encode_seq(seq1)
encoded_data2 = onehot_encode_seq(seq2)
encoded_data3 = onehot_encode_seq(seq3)
# %%
import tensorflow as tf
# via TF
def onehot_encode_seq_tf(input_seq):
    alphabet = {'A':0, 'T':1, 'C':2, 'G':3}
    indices = [alphabet[ch] for ch in input_seq]
    onehot_encoded = tf.one_hot(indices, 4, dtype=tf.uint8)
    return onehot_encoded
# %%
seq1= 'AAAATCGGC'
seq2 = 'TTGG'
seq3 = 'A'
out1 = onehot_encode_seq_tf(seq1)
out2 = onehot_encode_seq_tf(seq2)
out3 = onehot_encode_seq_tf(seq3)
# %%
# encode gene seq (e.g. aak-1) to a vector repr
# str to tf.tensor
import pandas as pd
import re
import pickle

# takes fp to dataset csv with 'Gene(s)' column, returns dict mapping gene to int
def gene_to_int_map(path):
    clean_sa_ds = pd.read_csv(path)
    d, idx = {}, 0
    for cur_genes in list(set(list(clean_sa_ds['Gene(s)']))):
        cur_genes = cur_genes.replace(" ","")
        genes = re.split(";|,",cur_genes)
        for gene in genes:
            if gene not in d:
                d[gene] = idx
                idx += 1
    return d

# save dict to pickle (because sets maybe? dont preserve relative ordering)
path = "../data/SynergyAge_DS/Cleaned_SynergyAge_Database.csv"
with open('./synergyage_gene_to_int_map.pickle', 'wb') as handle:
    pickle.dump(gene_to_int_map(path), handle, protocol=pickle.HIGHEST_PROTOCOL)

def genes_to_vec_tf(input_genes): # e.g. "sod-2;eat-2" --> tf.tensor([1,1,1,1,0,1,...])
    with open('./synergyage_gene_to_int_map.pickle', 'rb') as handle:
        alphabet = pickle.load(handle)
    cur_genes = input_genes.replace(" ","")
    genes = re.split(";|,",cur_genes)
    indices = [alphabet[ch] for ch in genes]
    vec = tf.constant([1 if i not in indices else 0 for i in range(len(alphabet.keys()))],
                      dtype='int32')
    return vec



# %%
synergy_age_ds = pd.read_csv(path);
synergy_age_ds_genes = synergy_age_ds['Gene(s)']
synergy_age_ds_genes = list(set(list(synergy_age_ds_genes)))
for i in range(100,110):
    
    vec_repr = genes_to_vec_tf(synergy_age_ds_genes[i])
    print(synergy_age_ds_genes[i]+'--->'+str(vec_repr.numpy()))