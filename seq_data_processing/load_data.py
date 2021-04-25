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
