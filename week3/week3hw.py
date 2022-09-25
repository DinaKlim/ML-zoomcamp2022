# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:52:13 2022

@author: Dina
"""

# Week 3 homework

import pandas as pd
import numpy as np

from scipy.stats import mode
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#%%

df = pd.read_csv('housing.csv')

#%%
# select columns and fill missing values with 0

features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value', 'ocean_proximity']

df = df[features]

df.fillna(0, inplace=True)

print(df.isnull().sum())

#%%
# Create a new column rooms_per_household by dividing the column total_rooms by the column households from dataframe.

df['rooms_per_household'] = df['total_rooms']/df['households']

# Create a new column bedrooms_per_room by dividing the column total_bedrooms by the column total_rooms from dataframe.

df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

# Create a new column population_per_household by dividing the column population by the column households from dataframe

df['population_per_household'] = df['population'] / df['households']

#%%

# What is the most frequent observation (mode) for the column ocean_proximity?

ocean_prox_mode = mode(df['ocean_proximity'])

print(ocean_prox_mode)

print(df['ocean_proximity'].value_counts())

#%% 

# correlation matrix

# select numerical features
print(df.dtypes)
numerical_columns = list(df.dtypes[df.dtypes != 'object'].index)


#%%
# correlation: Spearman because it's not linear, it's monotonic
sns.set(font_scale=1.5)
plt.figure(figsize=(15, 15))

df_corr = df[numerical_columns]

mask = np.triu(np.ones_like(df_corr.corr()))
fig1 = sns.heatmap(np.abs(df_corr.corr(method = 'pearson')), 
                   cmap='Reds', vmin=0, vmax=1, annot=True, mask=mask, fmt='.3g')
fig1.set_yticklabels(fig1.get_yticklabels(), rotation=0)
plt.savefig('Correlation_numerical.jpg', dpi = 300)

# highest correlation coefficient = 0.967 for households vs total_bedrooms
#%%
# create a variable above_average

df['above_average'] = (df.median_house_value > df.median_house_value.mean()).astype(int)

# Split your data in train/val/test sets, with 60%/20%/20% distribution.
# Use Scikit-Learn for that (the train_test_split function) and set the seed to 42.

np.random_seed = 42

# split the dataset into full train and test, 
# then split full train into train and validation

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)

# 20% out of 80% is 25%

df_train, df_val = train_test_split(df_full_train, test_size = 0.25)

# Make sure that the target value (median_house_value) is not in your dataframe

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.above_average.values
y_test = df_test.above_average.values
y_val = df_val.above_average.values

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

del df_train['above_average']
del df_val['above_average']
del df_test['above_average']

#%%

# Calculate the mutual information score with the (binarized) price for the categorical variable that we have. Use the training set only.
# What is the value of mutual information?
# Round it to 2 decimal digits using round(score, 2)

from sklearn.metrics import mutual_info_score

np.round(mutual_info_score(df.above_average, df.ocean_proximity), 2)

#%%

# Now let's train a logistic regression
# Remember that we have one categorical variable ocean_proximity in the data. Include it using one-hot encoding.

# we have deleted the median_house_value, so we need to use the new list of features

train_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

#%%
train_dicts = df_train[train_features].to_dict(orient='records')
val_dicts = df_val[train_features].to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_val = dv.fit_transform(val_dicts)
dv.get_feature_names_out()

#%%
# Fit the model on the training dataset.

from sklearn.linear_model import LogisticRegression

# To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
# model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)

#%%
model.fit(X_train, y_train)

#%%

y_pred = model.predict(X_val)

#%% 
# Calculate the accuracy on the validation dataset and round it to 2 decimal digits

y_pred = model.predict_proba(X_val)[:,1]
above_average_pred = (y_pred > 0.5)
accuracy = (y_val == above_average_pred).mean() # 0.84

#%% Q5
# exclude each feature from the model and train without the feature. Record the accuracy

for c in train_features:
    train_f = df_train.drop(c, axis=1)
    val_f = df_val.drop(c, axis=1)
    train_dicts = train_f.to_dict(orient='records')
    val_dicts = val_f.to_dict(orient='records')
    
    X_train_f = dv.fit_transform(train_dicts)
    X_val_f = dv.fit_transform(val_dicts)
    
    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_f, y_train)
    y_pred = model.predict_proba(X_val_f)[:,1]
    above_average_pred = (y_pred > 0.5)
    accuracy_f = (y_val == above_average_pred).mean()
    print('accuracy change without', c, accuracy - accuracy_f)    
    
#%% Q6

# use the original column 'median_house_value'. Apply the logarithmic transformation to this column.

# do the split again with the new y
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)

# 20% out of 80% is 25%

df_train, df_val = train_test_split(df_full_train, test_size = 0.25)

# Make sure that the target value (median_house_value) is not in your dataframe

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = np.log(df_train.median_house_value.values)
y_test = np.log(df_test.median_house_value.values)
y_val = np.log(df_val.median_house_value.values)

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

#%% 
train_dicts = df_train[train_features].to_dict(orient='records')
val_dicts = df_val[train_features].to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_val = dv.fit_transform(val_dicts)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

alphas = [0, 0.01, 0.1, 1, 10]
for a in alphas:
    model = Ridge(alpha=a, solver="sag", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse_alpha = np.round(mean_squared_error(y_val, y_pred), 3)
    print('rmse at alpha', a, rmse_alpha)
    