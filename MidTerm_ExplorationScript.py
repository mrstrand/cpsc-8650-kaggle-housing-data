
# %% [code] {"pycharm":{"is_executing":false}}
# -----------------------------------------------------
# This is a script version of MidTerm_Exploration.ipynb
# This was not run as a script but as a notebook in the ipynb format on kaggle
# This is just to be used as a way to read the logic that was preformed not for testing
# ---------------------------------------------------------


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.amp

# %% [code] {"pycharm":{"is_executing":false}}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
# % matplotlib inline

# %% [code] {"pycharm":{"is_executing":false}}
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

# %% [code] {"pycharm":{"is_executing":false}}
train.columns

# %% [code] {"pycharm":{"is_executing":false}}
train['SalePrice'].describe()

# %% [code] {"pycharm":{"is_executing":false}}
sns.distplot(train['SalePrice']);

# %% [code] {"pycharm":{"is_executing":false}}
# skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())

# %% [code] {"pycharm":{"is_executing":false}}
# scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));

# %% [code] {"pycharm":{"is_executing":false}}
# scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));

# %% [code] {"pycharm":{"is_executing":false}}
# box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

# %% [code] {"pycharm":{"is_executing":false}}
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

# %% [code] {"pycharm":{"is_executing":false}}
# correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

# %% [code] {"pycharm":{"is_executing":false}}
# saleprice correlation matrix
k = 10  # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

# %% [code] {"pycharm":{"is_executing":false}}
# super scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size=2.5)
plt.show();

# %% [code] {"pycharm":{"is_executing":false}}
# missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# %% [code]
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

# %% [code] {"pycharm":{"name":"#%%\n","is_executing":false}}
# see unique values of PoolQC
print('Unique values of PoolQC: ', train['PoolQC'].unique())


# %% [code] {"pycharm":{"name":"#%%\n","is_executing":false}}
# transform PoolQC column
def switch_pool_qc(arg):
    quality = {
        'Ex': 1,
        'Gd': 0.75,
        'TA': 0.5,
        'Fa': 0.25
    }
    return quality.get(arg, 0)


train['PoolQC'] = train['PoolQC'].transform(lambda x: switch_pool_qc(x))

# %% [code] {"pycharm":{"name":"#%%\n","is_executing":false}}
# see unique values of MiscFeature
print('Unique values of MiscFeature: ', train['MiscFeature'].unique())


# %% [code] {"pycharm":{"name":"#%%\n","is_executing":false}}
# transform MiscFeature column into multiple columns
def switch_pool_qc(arg):
    quality = {
        'Ex': 1,
        'Gd': 0.75,
        'TA': 0.5,
        'Fa': 0.25
    }
    return quality.get(arg, 0)


train['Shed'] = train['MiscFeature'].transform(lambda x: x == 'Shed' or 0)
train['SecondGarage'] = train['MiscFeature'].transform(lambda x: x == 'Gar2' or 0)
train['OtherMiscFeature'] = train['MiscFeature'].transform(lambda x: x == 'Othr' or 0)
train['TennisCourt'] = train['MiscFeature'].transform(lambda x: x == 'TenC' or 0)

# %% [code] {"pycharm":{"name":"#%%\n","is_executing":false}}
# see unique values of Alley
print('Unique values of Alley: ', train['Alley'].unique())


# %% [code] {"pycharm":{"name":"#%%\n","is_executing":false}}
# transform Alley column
def switch_alley(arg):
    quality = {
        'Grvl': 0.5,
        'Pave': 1,
    }
    return quality.get(arg, 0)


train['Alley'] = train['Alley'].transform(lambda x: switch_alley(x))

# %% [code] {"pycharm":{"name":"#%%\n","is_executing":false}}
# see unique values of Fence
print('Unique values of Fence: ', train['Fence'].unique())


# %% [code] {"pycharm":{"name":"#%%\n","is_executing":false}}
# transform Fence column
def switch_fence(arg):
    quality = {
        'GdPrv': 1,
        1: 1,
        'MnPrv': 0.75,
        0.75: 0.75,
        'GdWo': 0.5,
        0.5: 0.5,
        'MnWw': 0.25,
        0.25: 0.25
    }
    return quality.get(arg, 0)


train['Fence'] = train['Fence'].transform(lambda x: switch_fence(x))

# %% [code] {"pycharm":{"is_executing":false}}
# there is probably a better way than just deleting the data that has missing values
# for the pool we should fill in the missing values since NA means no pool so not really missing data here
# dealing with missing data
# A lot of the columns are like that.
# Misc feature could be turned into 0/1 for each feature
# Alley can be turned into 0/1 for paved alley, 0/1 for gravel alley
train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()  # just checking that there's no missing data missing...

# %% [code] {"pycharm":{"is_executing":false}}
# standardizing data
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:, np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# %% [code] {"pycharm":{"is_executing":false}}
# bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));

# %% [code] {"pycharm":{"is_executing":false}}
# deleting points
train.sort_values(by='GrLivArea', ascending=False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)

# %% [code] {"pycharm":{"is_executing":false}}
# bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));

# %% [code] {"pycharm":{"is_executing":false}}
# histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)

# %% [code] {"pycharm":{"is_executing":false}}
# applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])

# transformed histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)

# %% [code] {"pycharm":{"is_executing":false}}
# histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)

# %% [code] {"pycharm":{"is_executing":false}}
# data transformation
train['GrLivArea'] = np.log(train['GrLivArea'])

# transformed histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)

# %% [code] {"pycharm":{"is_executing":false}}
# histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)

# %% [code] {"pycharm":{"is_executing":false}}
# create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0
train.loc[train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
# transform data
train.loc[train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
# histogram and normal probability plot
sns.distplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)

# %% [code] {"pycharm":{"is_executing":false}}
# scatter plot
plt.scatter(train['GrLivArea'], train['SalePrice']);

# %% [code] {"pycharm":{"is_executing":false}}
# scatter plot
plt.scatter(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], train[train['TotalBsmtSF'] > 0]['SalePrice']);

# %% [code] {"pycharm":{"is_executing":false}}
# convert categorical variable into dummy
train = pd.get_dummies(train)

# %% [code] {"pycharm":{"is_executing":false}}
train.columns

# %% [code] {"pycharm":{"is_executing":false}}
train['SaleCondition_Family'].describe()

# %% [code]
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")