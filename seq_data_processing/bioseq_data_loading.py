# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:37:01 2021

@author: Bill Sun
"""

from Bio import SeqIO

path = '../data/Example_Animal_Genomes/anolis carolinensis (7.2 years max)/chrh.fna'
seqs= SeqIO.parse(open(path),'fasta')
# %%
for fasta in seqs:
    print(fasta.id)
    print(str(fasta.seq))
# %%
def fun(x):
    
    if x == 10: print('oh no!!')
    x -= 1
    x+= 100
    print(x)
# %%

test = [[1,2,3],[4,5,6],[7,8,9]]


def cumulativeSum(matrix, direction):
    if direction == 'row':
        return [sum(matrix[i]) for i in range(len(matrix))]
    elif direction == 'col':
        return [sum([matrix[i][j] for i in range(len(matrix))]) for j in range(len(matrix[0]))]
    
outRow = cumulativeSum(test, 'row')
outCol = cumulativeSum(test, 'col')