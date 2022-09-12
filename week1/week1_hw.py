# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:47:49 2022

@author: Dina
"""

import pandas as pd
import numpy as np


#%% read the dataset

df = pd.read_csv('data.csv')

#%% Question 2: Number of records in the dataset

num_records = len(df)

#%% Question 3: The most popular car manufacturers

df['Make'].value_counts()

# top three: Chevrolet, Ford, Volkswagen

#%% Question 4: Number of unique Audi car models

df_Audi = df[df['Make'] == 'Audi']

df_Audi_unique = len(df_Audi['Model'].value_counts())

#%% Question 5: Number of columns with missing values

num_cols_with_NaNs = len(df.columns[df.isnull().any()])

#%% Question 6: Does the median value change after filling missing values?

# Find the median value of "Engine Cylinders" column in the dataset
eng_cylinders_median = df['Engine Cylinders'].median()

# Next, calculate the most frequent value of the same "Engine Cylinders"
eng_cylinders_mode = df['Engine Cylinders'].mode()

# Use the fillna method to fill the missing values in "Engine Cylinders" 
# with the most frequent value from the previous step.

df_filled = df['Engine Cylinders'].fillna(4)

df_filled.to_csv('filled.csv')

# Now, calculate the median value of "Engine Cylinders" once again.
df_filled_median = df_filled.median()

# it did not change

#%% 
# Select all the "Lotus" cars from the dataset.
# Select only columns "Engine HP", "Engine Cylinders".

cols = ['Engine HP', 'Engine Cylinders']
df_q7 = df[df['Make'] == 'Lotus'][cols]

# Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 9 rows).
df_q7 = df_q7.drop_duplicates()

# Get the underlying NumPy array. Let's call it X.

X = df_q7.to_numpy()

# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
XT = X.transpose()
XTX = np.matmul(XT, X)

#%%

# Compute the inverse of XTX.

XTX_inverse = np.linalg.inv(XTX)

# Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800].

y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])

# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.

w = np.matmul(np.matmul(XTX_inverse, XT), y)
# What's the value of the first element of w? 4.59494481
