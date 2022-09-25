
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:00:05 2022

@author: Dina
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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