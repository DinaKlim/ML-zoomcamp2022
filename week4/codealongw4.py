
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:00:05 2022

@author: Dina
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.head().T)

#%%

df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

#%%
print(df.dtypes)

tc = pd.to_numeric(df.totalcharges, errors = 'coerce')

print(df[tc.isnull()][['customerid', 'totalcharges']])

df.totalcharges = df.totalcharges.fillna(0)

#%%
# replace the target value with 1
df.churn = (df.churn == 'yes').astype(int)

#%% setting up the validation framework

# split the dataset into full train and test, 
# then split full train into train and validation

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)

# 20% out of 80% is 25%

df_train, df_val = train_test_split(df_full_train, test_size = 0.25)

#%% 
print(len(df_train), len(df_val), len(df_test))

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.churn.values
y_test = df_test.churn.values
y_val = df_val.churn.values

#%%
del df_train['churn']
del df_val['churn']
del df_test['churn']

#%% EDA

df_full_train.reset_index(drop=True)

#%%
print(df_full_train.isnull().sum())

# distribution of variables in the target 
print(df_full_train.churn.value_counts(normalize = True))

# 11% is positive target

#%%

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

#%%
print(df_full_train[categorical].nunique())


#%%
global_churn = df_full_train.churn.mean()
global_churn
#%% Feature importance

# look at churn rates by different groups
from IPython.display import display
# for all cat variables
for c in categorical: 
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn

    display(df_group)
    print()
    print()
    
#%% mutual information
# mutual dependence
# how much we learn about churn if we learn from the variable

# the higher mutual info is, the more we learn

from sklearn.metrics import mutual_info_score

mutual_info_score(df_full_train.churn, df_full_train.contract)
#%%
mutual_info_score(df_full_train.contract, df_full_train.churn)

#%%
def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)
mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)

#%% one-hot encoding
from sklearn.feature_extraction import DictVectorizer

dicts = df_train[['gender', 'contract']].iloc[:10].to_dict(orient='records')

#%%

dv = DictVectorizer(sparse=False)
dv.fit(dicts)

#%%

dv.transform(dicts)

#%%
dv.get_feature_names()

#%% if the numerical value is included - it leaves it as is

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
train_dicts[0]

#%%
dv.fit(train_dicts)
dv.transform(train_dicts)

X_train= dv.fit_transform(train_dicts)
#%%
X_train.shape

#%% 
val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

#%%
# the difference between regression and logistic regression the 
# coefficients are between -1 and 1 with sigmoid
# input z
# sigmoid is 1 / 1 + exponent z
# from score to probability

# both regression and logistic regression are linear models

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

#%% 
print(model.coef_[0].round(3))
print(model.intercept_[0])

model.predict(X_train)

# these are hard predictions: we don't know the probability

y_pred = model.predict_proba(X_val)[:,1]

# soft predictions. The first column is the probability to be negative, 
# and the second column is the probability to be positive

#%% 
# make a hard decision: for the probability of above 0.5

churn_decision = (y_pred > 0.5)

#%%

# accuracy
accuracy = (y_val == churn_decision).mean()

#%%
df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
df_pred['correct'] = df_pred.prediction == df_pred.actual

print(df_pred['correct'].mean())

#%%

# week 4

from sklearn.metrics import accuracy_score
# accuracy_score(y_val, y_pred).mean()
accuracy_score(y_val, y_pred >= 0.5)

#%%
scores = []
threshold = np.linspace(0, 1, 21)
for t in threshold:
    
    score = accuracy_score(y_val, y_pred >t)
    print('%.2f %.3f' % (t, score))
    scores.append(score) 

#%%
plt.plot(threshold, scores)

#%%


#%%
from collections import Counter
Counter(y_pred >1)

#%%
# accuracy does not tell how good the model is
# class imbalance: dummy model predicting the majority class will already have a good accuracy
# for class imbalance, there are other ways to evaluate accuracy
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)

t=0.5 #threshold

predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

#%%
tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum() # false positive
fn = (predict_negative & actual_positive).sum() # false negative

# case where both predictive positive and actual positive are true

#%% confusion table is a 2x2 table
# churn predictions: churn - positive, not churn - negative
confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
    ])

confusion_matrix

# decide where to send the promotional email

# more false negatives than false positives
# instead of one number - 4 numbers with better understanding of how model performs

#%% 4.4 precision and recall

# true/ false positive / negative are used as metric
# precision
# fraction of correct positive predictions

# tp / total number of positive predictions
# tp / (tp + fp)

p = tp / (tp + fp)

# recall: fraction of correctly identified positive predictions
# churning customers

r = tp / (tp + fn)

#%% 4.5 ROC curves

# false positive rate, we want it to be smaller
# true positive rate, tpr, we want it to be bigger

tpr = tp / (tp + fn)

fpr = fp / (fp + tn)

# compute for all possible thresholds

thresholds = np.linspace(0, 1, 101)

scores = []
for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum() # false positive
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))
    
#%%
columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns = columns)

#%%
df_scores[::10] # print every 10th row

#%%
df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

#%%
plt.plot(df_scores.threshold, df_scores['tpr'], label = 'TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label = 'FPR')

# we want those curves to go down as fast as possible

#%% random model
y_rand = np.random.uniform(0, 1, size = len(y_val))
((y_rand >= 0.5) == y_val).mean()

def tpr_fpr_dataframe(y_val, y_pred):
    
    scores = []
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
         
         scores.append((t, tp, fp, fn, tn))
    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns = columns)
    
    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
     
    return df_scores
 
#%%
df_rand = tpr_fpr_dataframe(y_val, y_rand)

plt.plot(df_rand.threshold, df_rand['tpr'], label = 'TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label = 'FPR')
plt.legend()

#%%

# sort the predictions, set the threshold to maximize the accuracy
# ideal model predicts 100% of churning (above threshold) as churning

num_neg = (y_val ==0).sum()
num_pos = (y_val ==1).sum()

y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal_pred = np.linspace(0, 1, len(y_val))

#%%
((y_ideal_pred >= 0.726) == y_ideal).mean()

# this model does not exist but helps to benchmark

df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)

plt.plot(df_ideal.threshold, df_ideal['tpr'], label = 'TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label = 'FPR')
plt.legend()

# this model can identify churning customers correctly

#%%
# plot all benchmarks together
plt.plot(df_scores.threshold, df_scores['tpr'], label = 'TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label = 'FPR')

# plt.plot(df_rand.threshold, df_rand['tpr'], label = 'TPR')
# plt.plot(df_rand.threshold, df_rand['fpr'], label = 'FPR')

plt.plot(df_ideal.threshold, df_ideal['tpr'], label = 'TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label = 'FPR')
plt.legend()

# can plot the TP and FP

#%%

plt.plot(df_scores.fpr, df_scores.tpr, label = 'model')
# plt.plot(df_rand.fpr, df_rand.tpr, label = 'random')
plt.plot([0,1], [0,1])
plt.plot(df_ideal.fpr, df_ideal.tpr, label = 'ideal')

plt.xlabel('FPR')
plt.ylabel('TPR')

# ideal spot: as close as possible to the ideal, as far as possible from random
# model should not go below random baseline, we should flip positive and negative predictions

#%% sklearn implementation
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
# evaluates model at every point

#%% area under the curve - tells how good the classifier is
from sklearn.metrics import auc
# calculates area under any curve

auc(fpr, tpr)

auc(df_scores.fpr, df_scores.tpr)

#%% 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val, y_pred)

# auc - probability that randomly selected 
# how well our model can order customers and separate positive examples from negative


neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

import random

# randomly selected positive example

n=10000
success = 0
for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)
    
    if pos[pos_ind] > neg[neg_ind]:
        success = success +1
success / n

#%%
n=10000
pos_ind = np.random.randint(0,len(pos), size = n)
neg_ind = np.random.randint(0, len(neg), size = n)

#%%

(pos[pos_ind] > neg[neg_ind]).mean()

#%%

# 4.7 cross-validation
# parameter tuning
# select the best parameter

# split into train, val, test
# forget about test and use val to find best parameter

# for the val dataset, we can use it within the full train dataset

# split the training set into 3 parts
# train on 2 parts, val on 1 part
# use part number 2 for validation
# switch parts again

# will get 3 scores
# compute mean and standard deviation

# each cell is called "fold"

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model

dv, model = train(df_train, y_train, C=0.001)

#%%

def predict(df, dv, model):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    
    return y_pred

y_pred = predict(df_val, dv, model)

#%%
from sklearn.model_selection import KFold

kfold = KFold(n_splits = 5, shuffle = True, random_state = 1)

# use next method to see what is inside

train_idx, val_idx = next(kfold.split(df_full_train))

len(train_idx), len(val_idx)
len(df_full_train)

# repeat on 10 folds

#%%
from tqdm.auto import tqdm
n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
    
#%% 

dv, model = train(df_full_train, df_full_train.churn.values, C=1)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc