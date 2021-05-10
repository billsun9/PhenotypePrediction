# -*- coding: utf-8 -*-
"""
Created on Sat May  8 20:03:13 2021

@author: Bill Sun
"""

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_model(input_dim):
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=input_dim),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
        ])
    
    model.compile(optimizer='adam', loss='mean_absolute_error')
    
    return model


clean_sa_ds_path = '../../data/SynergyAge_DS/Cleaned_SynergyAge_Database.csv"'
data = pd.read_csv(clean_sa_ds_path)