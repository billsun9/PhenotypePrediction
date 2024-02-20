# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:08:51 2023

@author: Tony Yu
"""
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH = '../../Data/ChR/pnas.1700269114.sd01.csv'

df = pd.read_csv(PATH)


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
            print(set(ex['sequence']))
        
        if not is_number(ex['mKate_mean']):
            print(i, ex['mKate_mean'])
            badIdxs.add(i)
        if not is_number(ex['GFP_mean']):
            print(i, ex['GFP_mean'])
            badIdxs.add(i)
        if not is_number(ex['intensity_ratio_mean']):
            print(i, ex['intensity_ratio_mean'])
            badIdxs.add(i)
    return badIdxs


def clean(df):
    print(["Before Dropping"])
    badIdxs = check(df)
    df = df.drop(labels=badIdxs)
    df['sequence'] = df['sequence'].apply(lambda x: x.upper())
    df = df.reset_index(drop=True)
    df = df.rename(columns={"sequence": "Sequence", "GFP_mean": "Data"}, errors="raise")
    df['Sequence'].astype(str)
    df['Data'].astype(float)
    df['mKate_mean'].astype(float)
    # print(df.columns)
    return df[['Sequence','Data','mKate_mean']]


def visualize_distribution(df, num_bins=10):
    output = df['mKate_mean'].astype(float).to_numpy()
    print(output[:5])

    plt.hist(output, color='lightgreen', ec='black', density=True, label="mKate_mean", bins=num_bins)
    plt.legend()
    plt.title("Output Distribution")
    plt.xlabel("mKate_mean")
    plt.ylabel("Count")
    plt.show()


def visualize_bases(df):
    bases = ["A", "T", "C", "G"]
    input_data = df['Sequence'].to_numpy()
    bases_dict = defaultdict(lambda: 0)

    for sequence in input_data:
        for base in bases:
            bases_dict[base] += sequence.upper().count(base)

    names = list(bases_dict.keys())
    values = np.array(list(bases_dict.values()))

    plt.bar(range(len(bases_dict)), values / sum(values) * 100, tick_label=names)
    plt.title("Nucleotide Distribution")
    plt.xlabel("Nucleotide")
    plt.ylabel("% in all sequences")
    plt.show()


def z_score(data):
    mean = np.mean(data)
    sd = np.std(data)

    return (data - mean) / sd


def visualize_target_relations(df):
    mKate_mean = df['mKate_mean'].astype(float).to_numpy()
    GFP_mean = df['Data'].astype(float).to_numpy()

    mKate_mean = z_score(mKate_mean)
    GFP_mean = z_score(GFP_mean)

    plt.scatter(GFP_mean, mKate_mean)
    plt.legend()
    plt.title("mKate_mean vs GFP_mean")
    plt.xlabel("GFP_mean")
    plt.ylabel("mKate_mean")
    plt.show()


def one_hot_encode(sequences):
    nucleotide_mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    num_sequences = len(sequences)
    max_sequence_length = 0

    for sequence in sequences:
        max_sequence_length = max(max_sequence_length, len(sequence))

    # Initialize an array of zeros for one-hot encoding
    encoding = np.zeros((num_sequences, len(nucleotide_mapping), max_sequence_length), dtype=int)

    # Iterate through the sequence and set the corresponding index to 1
    for i, sequence in enumerate(sequences):
        for j, nucleotide in enumerate(sequence):
            if nucleotide in nucleotide_mapping:
                encoding[i, nucleotide_mapping[nucleotide], j] = 1
            else:
                raise ValueError(f"Invalid nucleotide: {nucleotide}")

    return encoding


def apply_one_hot_and_pca(df):
    sequences = df["Sequence"].to_numpy()

    encoding = one_hot_encode(sequences)

    data_2d = np.array([features_2d.flatten() for features_2d in encoding])

    pca_agent = PCA(n_components=5)
    pca_agent.fit(data_2d)

    eigv = np.real(pca_agent.eigenvalues)

    names = [f"PC {i + 1}" for i in range(len(eigv))]

    plt.bar(range(len(eigv)), eigv / sum(eigv) * 100, tick_label=names)
    plt.title("PCA Result")
    plt.xlabel("PC #")
    plt.ylabel("% Variance")
    plt.show()

    transformed_data = np.real(pca_agent.transform(data_2d))

    plt.scatter(transformed_data[: 100, 0], transformed_data[:100, 1], s=20)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.eigenvalues = None

    def fit(self, X):
        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0)
        centered_data = X - self.mean

        # Calculate the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors as principal components
        self.components = eigenvectors[:, :self.n_components]
        self.eigenvalues = eigenvalues[:self.n_components]

    def transform(self, X, dimension=2):
        if self.mean is None or self.components is None:
            raise ValueError("PCA not fitted. Call fit() method first.")

        # Center the data and project it onto the principal components
        centered_data = X - self.mean
        transformed_data = centered_data @ self.components[:, : dimension]

        return transformed_data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


df = clean(df)

visualize_distribution(df)
visualize_bases(df)
apply_one_hot_and_pca(df)

visualize_target_relations(df)
