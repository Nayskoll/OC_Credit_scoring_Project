#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

import os
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')
import pickle 
import matplotlib.pyplot as plt
import seaborn as sns


# # Preparing test dataset

# In[30]:


app_train = pd.read_csv("application_train.csv")
app_test_raw = pd.read_csv("application_test.csv")
app_test = pd.read_csv("application_test.csv")
schema_list = pd.read_csv("schema_for_data_prep.csv").columns.tolist()


# In[31]:


le = LabelEncoder()
le_count = 0

# encoding to 0/1 when there are less than 3 values 
# example: gender will take the value of 0/1 instead of M/F
for col in app_train:
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            le_count += 1
print('%d columns were label encoded.' % le_count)


# In[32]:


app_test = pd.get_dummies(app_test)

print('Testing Features shape: ', app_test.shape)

# test set has more features.


# In[33]:


app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == app_test['DAYS_EMPLOYED'].max()
app_test['DAYS_EMPLOYED'].replace({app_test['DAYS_EMPLOYED'].max(): np.nan}, inplace = True)
print('There are %d anomalies in the test data out of %d entries' % (app_test['DAYS_EMPLOYED_ANOM'].sum(), len(app_test)))


# In this method, we generate powers or interaction terms of existing features as new features.

# In[34]:


poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

imputer = SimpleImputer(strategy = 'median')

poly_features_test = imputer.fit_transform(poly_features_test)


poly_transformer = PolynomialFeatures(degree = 3)


# In[35]:


poly_features_test = poly_transformer.fit_transform(poly_features_test)
print('Polynomial Features shape: ', poly_features_test.shape)


# In[36]:


poly_features_test = pd.DataFrame(poly_features_test, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = ['SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'], how = 'left')

print('Testing data with polynomial features shape: ', app_test_poly.shape)


# In[37]:


app_test_ready = app_test_poly[schema_list]
#necessary because of dummies values


# In[38]:


from sklearn.preprocessing import MinMaxScaler

features = app_test_ready.columns.tolist()

test = app_test_ready.copy()

imputer = SimpleImputer(strategy = 'median')

scaler = MinMaxScaler(feature_range = (0,1))

imputer.fit(test)

test = imputer.transform(test)

scaler.fit(test)
test = scaler.transform(test)

print('Testing data shape: ', test.shape)


# # Loading the model

# In[39]:


filename = 'logreg_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


# In[ ]:





# # Getting the features coefficient

# In[40]:


features_coeff = pd.DataFrame(features, index=None, columns=["features"])
features_coeff["importance"] = loaded_model.coef_[0]
features_coeff["importance_abs"] = features_coeff["importance"].abs()


# In[41]:


# removing polynomial features from features importance (as it is harder to explain to a client)
features_coeff_without_poly = features_coeff[features_coeff["features"].isin(app_test.columns.tolist())]
features_coeff_without_poly = features_coeff_without_poly.sort_values("importance_abs", ascending=False)


# In[42]:


features_coeff_without_poly["rank"] = features_coeff_without_poly["importance_abs"].rank(ascending=False).astype(int)


# In[43]:


features_coeff_without_poly.head()


# In[1]:


features_coeff_without_poly.to_csv("features_importance.csv", index=False, sep=",")


# # Applying the model to the test dataset

# In[45]:


result = loaded_model.predict_proba(test)[:, 1]


# ## With scaled data

# In[46]:


app_test["% default"] = result * 100
app_test["Result"] = (result >= 0.12).astype(bool)
app_test = app_test.set_index("SK_ID_CURR")
app_test.to_csv("client_list.csv", sep=",")


# ## With raw data

# In[47]:


app_test_raw["% default"] = result * 100
app_test_raw = app_test_raw.set_index("SK_ID_CURR")
app_test_raw.to_csv("client_list_raw.csv", sep=",")


# # Preparing dataset for radar chart vizualisation for each user
# We'll apply a decile classification

# In[23]:


no_default_users = app_test[app_test.Result==False].index.tolist()
default_users = app_test[app_test.Result==True].index.tolist()


# In[24]:


top_10_features = features_coeff_without_poly.head(10).features.values


# In[25]:


scaled_data = pd.DataFrame(test, columns=app_test_ready.columns.tolist())[top_10_features]


# In[26]:


scaled_data['SK_ID_CURR'] = app_test.index
scaled_data = scaled_data.set_index("SK_ID_CURR")


# In[27]:


for ft in top_10_features:                                   
    scaled_data[ft] = pd.qcut(scaled_data[ft].values, q=10, duplicates="drop").codes + 1


# In[28]:


scaled_data.to_csv("scaled_data.csv", index="SK_ID_CURR", sep=",")


# # Testing the vizualisation before putting in on streamlit

# In[365]:


scaled_data = scaled_data.reset_index()


# In[366]:


radiar_chart_default = scaled_data[scaled_data["SK_ID_CURR"].isin(default_users)].drop(columns="SK_ID_CURR").mean()
default_df = pd.DataFrame(radiar_chart_default, columns=["values"]).reset_index()

radiar_chart_no_default = scaled_data[scaled_data["SK_ID_CURR"].isin(no_default_users)].drop(columns="SK_ID_CURR").mean()
no_default_df = pd.DataFrame(radiar_chart_no_default, columns=["values"]).reset_index()


# In[367]:


one_client_df = scaled_data[scaled_data["SK_ID_CURR"]==100042].drop(columns="SK_ID_CURR").T.reset_index().rename(columns={0:"values"})


# In[368]:


one_client_df.columns=["index", "values"]


# In[369]:



fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=default_df["values"],
      theta=default_df["index"],
      fill='toself',
      name='Default mean',
      mode = 'lines',
      line_color = 'Red'))

fig.add_trace(go.Scatterpolar(
      r=no_default_df["values"],
      theta=no_default_df["index"],
      fill='toself',
      name='No Default Mean',
      mode = 'lines',
      line_color = 'Green'
))

fig.add_trace(go.Scatterpolar(
      r=one_client_df["values"],
      theta=one_client_df["index"],
      fill='toself',
      name='Client Selected',
      mode = 'lines',
      line_color = 'Blue'
))


fig.show()

