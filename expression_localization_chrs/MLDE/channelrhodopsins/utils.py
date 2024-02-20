import os   

class NucleotideToAA:
    def __init__(self):

        self.codon_map = {
            'TCA': 'S',    # Serina
            'TCC': 'S',    # Serina
            'TCG': 'S',    # Serina
            'TCT': 'S',    # Serina
            'TTC': 'F',    # Fenilalanina
            'TTT': 'F',    # Fenilalanina
            'TTA': 'L',    # Leucina
            'TTG': 'L',    # Leucina
            'TAC': 'Y',    # Tirosina
            'TAT': 'Y',    # Tirosina
            'TAA': '*',    # Stop
            'TAG': '*',    # Stop
            'TGC': 'C',    # Cisteina
            'TGT': 'C',    # Cisteina
            'TGA': '*',    # Stop
            'TGG': 'W',    # Triptofano
            'CTA': 'L',    # Leucina
            'CTC': 'L',    # Leucina
            'CTG': 'L',    # Leucina
            'CTT': 'L',    # Leucina
            'CCA': 'P',    # Prolina
            'CCC': 'P',    # Prolina
            'CCG': 'P',    # Prolina
            'CCT': 'P',    # Prolina
            'CAC': 'H',    # Histidina
            'CAT': 'H',    # Histidina
            'CAA': 'Q',    # Glutamina
            'CAG': 'Q',    # Glutamina
            'CGA': 'R',    # Arginina
            'CGC': 'R',    # Arginina
            'CGG': 'R',    # Arginina
            'CGT': 'R',    # Arginina
            'ATA': 'I',    # Isoleucina
            'ATC': 'I',    # Isoleucina
            'ATT': 'I',    # Isoleucina
            'ATG': 'M',    # Methionina
            'ACA': 'T',    # Treonina
            'ACC': 'T',    # Treonina
            'ACG': 'T',    # Treonina
            'ACT': 'T',    # Treonina
            'AAC': 'N',    # Asparagina
            'AAT': 'N',    # Asparagina
            'AAA': 'K',    # Lisina
            'AAG': 'K',    # Lisina
            'AGC': 'S',    # Serina
            'AGT': 'S',    # Serina
            'AGA': 'R',    # Arginina
            'AGG': 'R',    # Arginina
            'GTA': 'V',    # Valina
            'GTC': 'V',    # Valina
            'GTG': 'V',    # Valina
            'GTT': 'V',    # Valina
            'GCA': 'A',    # Alanina
            'GCC': 'A',    # Alanina
            'GCG': 'A',    # Alanina
            'GCT': 'A',    # Alanina
            'GAC': 'D',    # Acido Aspartico
            'GAT': 'D',    # Acido Aspartico
            'GAA': 'E',    # Acido Glutamico
            'GAG': 'E',    # Acido Glutamico
            'GGA': 'G',    # Glicina
            'GGC': 'G',    # Glicina
            'GGG': 'G',    # Glicina
            'GGT': 'G'     # Glicina
        }

    def translate_dna_to_amino_acids(self, dna_sequence):
        if len(dna_sequence) % 3 != 0:
            raise ValueError("Length of the DNA sequence must be a multiple of 3.")
    
        amino_acids = []
    
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i:i+3]
            if codon not in self.codon_map: raise ValueError("{} is invalid codon".format(codon))
            amino_acid = self.codon_map[codon]
            if amino_acid == '*': break
            amino_acids.append(amino_acid)
    
        return ''.join(amino_acids)
    
    # Example usage:
    def ex(self):
        dna_sequence = "TTTTAAGATGAT"
        amino_acids_result = self.translate_dna_to_amino_acids(dna_sequence)
        print("DNA sequence:", dna_sequence)
        print("Amino acids:", amino_acids_result)
# %%
import collections
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class CGR():
    def __init__(self): pass

    def count_kmers(self, sequence, k):
        d = collections.defaultdict(int)
        for i in range(len(sequence) - (k - 1)):
            d[sequence[i:i + k]] += 1
        return d

    def probabilities(self, sequence, kmer_count, k):
        probabilities = collections.defaultdict(float)
        N = len(sequence)
        for key, value in kmer_count.items():
            probabilities[key] = float(value) / (N - k + 1)
        return probabilities

    def chaos_game_representation(self, probabilities, k):
        array_size = int(math.sqrt(4 ** k))
        chaos = []
        for i in range(array_size):
            chaos.append([0] * array_size)
        maxx = array_size
        maxy = array_size
        posx = 1
        posy = 1
        for key, value in probabilities.items():
            for char in key:
                if char == "T":
                    posx += maxx / 2
                elif char == "C":
                    posy += maxy / 2
                elif char == "G":
                    posx += maxx / 2
                    posy += maxy / 2
                maxx /= 2
                maxy /= 2

            chaos[int(posy - 1)][int(posx - 1)] = value
            maxx = array_size
            maxy = array_size
            posx = 1
            posy = 1
        m = float(np.amax(chaos))
        c = np.array(chaos) / m
        return c

    def generate_cgr_from_sequence(self, sequence, k, fp=None):
        kmers = self.count_kmers(sequence, k)
        kmers_prob = self.probabilities(sequence, kmers, k)
        cgr_output = self.chaos_game_representation(kmers_prob, k)
        
        if fp:
            plt.figure(figsize=(12, 12))
        
            plt.imshow(cgr_output, cmap=cm.gray_r)
            plt.axis('off')
            plt.savefig(fp)
            # plt.show()
    
        return cgr_output

# takes list of nucleotide sequences; potentially saves to save directory
def makeCGRs(seqs, k=5, save_dir=""):
    cgr = CGR()
    res = []
    for i, seq in enumerate(seqs):
        if save_dir:
            save_fp = os.path.join(save_dir, "{}_k={}.png".format(i,k))
            output = cgr.generate_cgr_from_sequence(seq, k, fp=save_fp)
        else:
            output = cgr.generate_cgr_from_sequence(seq, k)
        res.append(output)
    return res