# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 19:10:16 2022

@author: Dina
"""

# homework week 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#%%
# read the dataset
df = pd.read_csv('AER_credit_card_data.csv')

#%% data preparation
# Create the target variable by mapping yes to 1 and no to 0.
dict_card = {'yes': 1, 'no': 0}
df['target'] = df.card.map(dict_card)

#%%
# Split the dataset into 3 parts: train/validation/test with 60%/20%/20% distribution. 
# Use train_test_split funciton for that with random_state=1

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=1)

len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values

del df_train['target']
del df_val['target']
del df_test['target']
#%%
from sklearn.metrics import roc_auc_score

# For each numerical variable, use it as score and compute AUC with the card variable.
# Use the training dataset for that.

numerical = ['reports', 'age', 'income', 'share', 'expenditure', 
             'dependents', 'months', 'majorcards', 'active']


#%% Question 1
for col in numerical:
    auc = roc_auc_score(y_train, df_train[col])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[col])
    print('%9s, %.3f' % (col, auc))

# max auc for "dependents"

#%% Question 2

features = ["reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active", "owner", "selfemp"]

#%% one-hot encoding
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
train_dicts = df_train[features].to_dict(orient='records')

X_train=dv.fit_transform(train_dicts)

#%%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)

model.fit(X_train, y_train)

#%%
val_dicts = df_val[features].to_dict(orient='records')
X_val=dv.fit_transform(val_dicts)
X_val=dv.fit_transform(val_dicts)

#%%
y_pred = model.predict_proba(X_val)[:, 1]

model_score = roc_auc_score(y_val, y_pred)
print ('Logistic regression score:', '%.3f' % model_score)

#%% Question 3

# Evaluate the model on the validation dataset on all thresholds from 0.0 to 1.0 with step 0.01
# For each threshold, compute precision and recall
# Plot them
metric = []

thresholds = np.linspace(0, 1, 101)
for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum() # false positive
    fn = (predict_negative & actual_positive).sum()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    metric.append((t, precision, recall))
    
columns = ['threshold', 'precision', 'recall']
df_scores = pd.DataFrame(metric, columns = columns)

#%%
plt.plot(df_scores['threshold'], df_scores['precision'])
plt.plot(df_scores['threshold'], df_scores['recall'])

# the intersection is at 0.3

#%% Question 4

f1_score = []

thresholds = np.linspace(0, 1, 101)
for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum() # false positive
    fn = (predict_negative & actual_positive).sum()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1 = 2 * precision * recall / (precision + recall)
    f1_score.append((t, f1))
    
columns_f1 = ['threshold', 'f1_score']
df_f1_score = pd.DataFrame(f1_score, columns = columns_f1)

print(df_f1_score[df_f1_score.f1_score == df_f1_score.f1_score.max()])

#%% Question 4
from sklearn.model_selection import KFold

# functions for training and predicting with set parameters for Logistic Regression 


def train(df_train, y_train, C=1.0):
    
    dicts = df_train[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[features].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

#%%
scores = []
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.target.values
    y_val = df_val.target.values

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

#%% Question 6

# initialize C
for C in [0.01, 0.1, 1, 10]:
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.target.values
        y_val = df_val.target.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# ties: 1 and 10, equal std, choosing the smallest C