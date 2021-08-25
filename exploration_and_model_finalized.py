#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.metrics import roc_curve, plot_roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, f1_score

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import pyplot
import lightgbm as lgb
import gc
import os

import warnings
warnings.filterwarnings('ignore')
import pickle


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


app_train = pd.read_csv("application_train.csv")
print('Training data shape : ', app_train.shape)
app_train.head()


# In[4]:


app_train.TARGET.mean()
#8% of default


# In[5]:


app_test = pd.read_csv("application_test.csv")
print('Testing data shape: ', app_test.shape)
app_test.head()


# In[6]:


app_train['TARGET'].value_counts()


# In[7]:


app_train['TARGET'].astype(int).plot.hist()


# In[8]:


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis = 1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0:"Missing Values", 1:"% of Total Values"})
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending = False).round(1)
    
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
    
    return mis_val_table_ren_columns


# In[9]:


missing_values = missing_values_table(app_train)
missing_values.head(20)


# In[10]:


app_train.dtypes.value_counts()


# In[11]:


app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[12]:


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


# In[13]:


app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# test set has more features.


# In[112]:


app_test.head()


# In[15]:


train_labels = app_train['TARGET']

app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# In[16]:


(app_train['DAYS_BIRTH'] / -365).describe()


# In[17]:


app_train['DAYS_EMPLOYED'].describe()


# In[18]:


app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')


# In[19]:


plt.boxplot(app_train['DAYS_EMPLOYED'])


# In[20]:


anom = app_train[app_train['DAYS_EMPLOYED'] == app_train['DAYS_EMPLOYED'].max()]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != app_train['DAYS_EMPLOYED'].max()]
print('The non_anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))


# In[21]:


app_train['DAYS_EMPLOYED_ANOM'] = app_train['DAYS_EMPLOYED'] == app_train['DAYS_EMPLOYED'].max()
app_train['DAYS_EMPLOYED'].replace({app_train['DAYS_EMPLOYED'].max(): np.nan}, inplace = True)
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')


# In[22]:


app_test['DAYS_EMPLOYED_ANOM'] = app_test['DAYS_EMPLOYED'] == app_test['DAYS_EMPLOYED'].max()
app_test['DAYS_EMPLOYED'].replace({app_test['DAYS_EMPLOYED'].max(): np.nan}, inplace = True)
print('There are %d anomalies in the test data out of %d entries' % (app_test['DAYS_EMPLOYED_ANOM'].sum(), len(app_test)))


# In[23]:


correlations = app_train.corr()['TARGET'].sort_values()

print('Most Positive Correlations:\n', correlations.tail(15))
print('Most Negative Correlations:\n', correlations.head(15))


# In[24]:


app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])


# In[25]:


plt.style.use('fivethirtyeight')

plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client')
plt.xlabel('Age (years)')
plt.ylabel('Count')


# In[26]:


plt.figure(figsize = (10, 8))
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Ages')


# In[27]:


age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)


# In[28]:


age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups


# In[29]:


plt.figure(figsize = (8,8))
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
plt.xticks(rotation = 75)
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')


# In[30]:


ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs


# In[31]:


plt.figure(figsize = (8, 6))
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')


# In[32]:


plt.figure(figsize = (10, 12))

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    plt.subplot(3, 1, i+1)
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source)
    plt.ylabel('Density')
    plt.legend()
    
plt.tight_layout(h_pad = 2.5)


# In[33]:


plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()
plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

plot_data = plot_data.dropna().loc[:100000, :]

def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate('r = {:.2f}'.format(r), xy = (.2, .8), xycoords = ax.transAxes, size = 20)
    
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey = False, hue = 'TARGET', vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

grid.map_upper(plt.scatter, alpha = 0.2)
grid.map_diag(sns.kdeplot)
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05)


# ## In this method, we generate powers or interaction terms of existing features as new features.

# In[34]:


poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]



imputer = SimpleImputer(strategy = 'median')

poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])

poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)


poly_transformer = PolynomialFeatures(degree = 3)


# In[35]:


poly_transformer.fit(poly_features)

poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)


# In[36]:


poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]


# In[37]:


poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

poly_features['TARGET'] = poly_target

poly_corrs = poly_features.corr()['TARGET'].sort_values()

print(poly_corrs.head(10))
print(poly_corrs.tail())


# In[38]:


poly_features_test = pd.DataFrame(poly_features_test, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
app_train_poly = app_train.merge(poly_features, on = ['SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'], how = 'left')
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
app_test_poly = app_test.merge(poly_features_test, on = ['SK_ID_CURR', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'], how = 'left')

app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join = 'inner', axis = 1)
print('Training data with polynomial features shape: ', app_train_poly.shape)
print('Testing data with polynomial features shape: ', app_test_poly.shape)


# In[40]:


# adding new features


# In[41]:


app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']


# In[42]:


app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']


# In[43]:


plt.figure(figsize = (12, 20))

for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    plt.subplot(4, 1, i+1)
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label = 'target == 0')
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label = 'target == 1')
    plt.title('Distribution of %s by Target Value' % feature)
    plt.xlabel('%s' % feature)
    plt.ylabel('Density')
    plt.legend()
    
plt.tight_layout(h_pad = 2.5)


# In[45]:



if 'TARGET' in app_train_poly:
    train = app_train_poly.drop(columns = ['TARGET'])
else:
    train = app_train_poly.copy()
    
features = list(train.columns)

test = app_test_poly.copy()

imputer = SimpleImputer(strategy = 'median')

scaler = MinMaxScaler(feature_range = (0,1))

imputer.fit(train)

train = imputer.transform(train)
test = imputer.transform(test)

scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)


# # Prediction models

# In[46]:


# Grid search cross validation

X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.3, random_state=0)

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train, y_train)


# In[115]:


logreg_cv.best_params_


# ## Applying a logistic regression with the best parameters
# using only the given train dataset as we don't have labels on the target dataset

# In[47]:


logreg_optimal=LogisticRegression(penalty='l2', C=10)

logreg_optimal.fit(X_train, y_train)


# In[100]:


plot_roc_curve(logreg_optimal, X_test, y_test)  
plt.show() 


# Here we can see that, if we want to catch at least 50% of True positive, we have to accept that 20% of the positive results will be wrong (false positive)

# In[118]:


y_predict_lr = logreg_optimal.predict_proba(X_test)[:,1]

result = (y_predict_lr >= 0.5).astype(bool)

for i in np.arange(0.1, 0.4, 0.02):
    result = (y_predict_lr >= i).astype(bool)
    print(i, "as a threshold")
    print("recall_score", recall_score(y_test, result))
    print("precision_score", precision_score(y_test, result))
    print("f1_score", f1_score(y_test, result))
    print("roc_auc_score", roc_auc_score(y_test, result))
    print("---")


# We'll consider that the recall score has to be > 50%. We'll use 0.12 as a rounding value. It's also one of the best values for f1 score.

# In[116]:


def plot_confusion_matrix(target, prediction):
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_matrix(target, prediction.round()), center=0,
                cmap=sns.color_palette("BrBG", 7), linewidths=1,
                annot=True, annot_kws={"size": 10}, fmt=".02f")

    plt.title("Matrix confusion", fontsize=18)
    plt.ylabel('TARGET', fontsize=13)
    plt.xlabel('PREDICTION', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(ticks=plt.yticks()[0], rotation=0)

    plt.show()
    
    target_names = ['class 0', 'class 1']
    print(classification_report(target, prediction.round(), target_names=target_names))


# In[94]:


result = (y_predict_lr >= 0.12).astype(bool)

plot_confusion_matrix(y_test, result)


# We predict 3711 / (3711+3629) = 51% of the default payment. Only 20% of our positive predictions are accurate (precision)

# In[101]:


lr_auc = roc_auc_score(y_test, result)
print('ROC AUC=%.3f' % (lr_auc))


# ## Save and export model

# In[88]:


# get the right columns
app_test_poly.to_csv('schema_for_data_prep.csv', index = False)

filename = 'logreg_model.sav'
pickle.dump(logreg_optimal, open(filename, 'wb'))


# # Testing random forrest classifier
# We've tested a random forest classifier, which gave similar results. The logistic regression is easier to understand.

# In[107]:


random_forest_poly = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
random_forest_poly.fit(X_train, y_train)

X_test_predict_rf = random_forest_poly.predict_proba(X_test)[:,1]


# In[106]:


y_predict_lr = logreg_optimal.predict_proba(X_test)[:,1]

result = (y_predict_lr >= 0.5).astype(bool)

for i in np.arange(0.1, 0.4, 0.02):
    result = (y_predict_lr >= i).astype(bool)
    print(i, "as a threshold")
    print("recall_score", recall_score(y_test, result))
    print("precision_score", precision_score(y_test, result))
    print("f1_score", f1_score(y_test, result))
    print("roc_auc_score", roc_auc_score(y_test, result))
    print("---")


# Results are very similar but the logistic regression is easier to understand

# In[108]:


result = (X_test_predict_rf >= 0.12).astype(bool)

roc_auc_score(y_test, result)


# In[109]:


plot_confusion_matrix(y_test, result)

