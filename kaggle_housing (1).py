#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join
import pandas as pd
import numpy as np
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt
import seaborn as sns



# # Dataset

# In[ ]:


data_dir = os.getenv('HOME')+'/aiffel/kaggle_kakr_housing/data'

train_data_path = join(data_dir, 'train.csv')
test_data_path = join(data_dir, 'test.csv') 

train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)


# In[ ]:


train.head()


# In[ ]:


train['date'] = train['date'].apply(lambda i: i[:6]).astype(int)
del train['id']
train.head()


# In[ ]:


y = train['price']
del train['price']


# In[ ]:


del train['id']


# In[ ]:


test['date'] = test['date'].apply(lambda i: i[:6]).astype(int)

del test['id']
print(test.columns)


# In[ ]:


y


# In[ ]:


sns.kdeplot(y)
plt.show()


# In[ ]:


y = np.log1p(y)
y


# In[ ]:


sns.kdeplot(y)
plt.show()


# In[ ]:


train.info()


# # GridSearch

# In[ ]:


def GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5):
    grid_model = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=5)
    grid_model.fit(train, y)
    
    params = grid_model.cv_results_['params']
    score = grid_model.cv_results_['mean_test_score']
    
    results = pd.DataFrame(params)
    results['score'] = score
    results['RMSLE'] = np.sqrt(-1 * results['score'])
    results = results.sort_values('RMSLE')
    
    return results


# # LGBMRegressor GridSearch
# 

# In[ ]:



random_state = 2020
param_grid = {
    'objective':['regression'],
    'learning_rate' : [0.05],
    'n_estimators':[100,500,1000],
    'max_depth': [1, 10, 20, 30, 40, 50],
}

model = LGBMRegressor(random_state=random_state)
GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5)


# # XGBRegressor GridSearch

# In[ ]:


param_grid = {
    'learning_rate': [0.05],
    'subsample': [0.9],
    'n_estimators':[100,500,1000],
    'max_depth': [1, 5, 10, 20, 30, 40, 50],
}

model = XGBRegressor(random_state=random_state)
GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5)


# # GradientBoostingRegressor GridSearch

# In[ ]:


param_grid = {
    'learning_rate': [0.05],
    'subsample': [0.9],
    'n_estimators':[100,500,1000],
    'max_depth': [1, 5, 10]
}

model = GradientBoostingRegressor(random_state=random_state)
GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5)


# # RandomForestRegressor GridSearch

# In[ ]:


param_grid = {
    'n_estimators':[100,500,1000],
    'max_depth': [1, 5, 10, 20, 30, 40, 50],
}

model = RandomForestRegressor(random_state=random_state)
GridSearch(model, train, y, param_grid, verbose=2, n_jobs=5)


# # Train

# In[ ]:



LGB_model = LGBMRegressor(
    max_depth=10, 
    n_estimators=1000, 
    learning_rate=0.05, 
    objective='regression', 
    random_state=random_state)

LGB_model.fit(train, y)
LGB_prediction = LGB_model.predict(test)
LGB_prediction = np.expm1(LGB_prediction)


# In[ ]:


XGB_model = XGBRegressor(
    max_depth=5, 
    n_estimators=1000, 
    learning_rate=0.05, 
    subsample=0.9, 
    random_state=random_state)

XGB_model.fit(train, y)
XGB_prediction = XGB_model.predict(test)
XGB_prediction = np.expm1(XGB_prediction)


# In[ ]:


GBR_model = GradientBoostingRegressor(
    max_depth=5, 
    n_estimators=1000, 
    learning_rate=0.05, 
    subsample=0.9, 
    random_state=random_state)

GBR_model.fit(train, y)
GBR_prediction = GBR_model.predict(test)
GBR_prediction = np.expm1(GBR_prediction)


# In[ ]:


RFR_model = RandomForestRegressor(
    max_depth=50, 
    n_estimators=1000, 
    random_state=random_state)

RFR_model.fit(train, y)
RFR_prediction = RFR_model.predict(test)
RFR_prediction = np.expm1(RFR_prediction)


# In[ ]:


prediction = 0.25* XGB_prediction + 0.25*LGB_prediction + 0.25*RFR_prediction +0.25*GBR_prediction


# # Submission

# In[ ]:


data_dir = os.getenv('HOME')+'/aiffel/kaggle_kakr_housing/data'

submission_path = join(data_dir, 'sample_submission.csv')
submission = pd.read_csv(submission_path)

submission['price'] = prediction

submission_csv_path = '{}/submission_{}.csv'.format(data_dir, 'LGB_XGB_GBR_RFR')
submission.to_csv(submission_csv_path, index=False)


# In[ ]:


submission.head()


# In[ ]:





# In[ ]:





# In[ ]:




