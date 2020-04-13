# %% [code]

#The script version of the kaggle notebook
#It is easier to work this way then copy and paste it into kaggle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
# uncomment this line when running on kaggle
# % matplotlib inline

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", na_values=['?', ''], delimiter=',',
                 delim_whitespace=False)
data = df

# %% [code]
# get the index of the outliers in GrLivArea
# the paper reccomends removing these
get_outliers = data.sort_values(by="GrLivArea", ascending=False).head(2)
get_outliers

# %% [code]
data.drop([523, 1298], inplace=True)
data.reset_index(inplace=True)

# %% [code]

print("Shape of training set: ", df.shape)
print("Missing values before remove NA: ")
print(data.columns[data.isnull().any()])

# %% [code]
# Fields that used NULL for absence of feature on the house
data.Alley.fillna(inplace=True, value='No')
data.BsmtCond.fillna(inplace=True, value='No')
data.BsmtExposure.fillna(inplace=True, value='No')
data.BsmtFinType1.fillna(inplace=True, value='No')
data.BsmtFinType2.fillna(inplace=True, value='No')
data.BsmtQual.fillna(inplace=True, value='No')
data.Fence.fillna(inplace=True, value='No')
data.FireplaceQu.fillna(inplace=True, value='No')
data.GarageCond.fillna(inplace=True, value='No')
data.GarageFinish.fillna(inplace=True, value='No')
data.GarageQual.fillna(inplace=True, value='No')
data.GarageType.fillna(inplace=True, value='No')
data.MiscFeature.fillna(inplace=True, value='No')
data.PoolQC.fillna(inplace=True, value='No')

print("Missing values after insert No, i.e., real missing values: ")
print(data.columns[data.isnull().any()])

# %% [code]

# Numerical just insert 0s in the nulls
data.BsmtFinSF1.fillna(inplace=True, value=0)
data.BsmtFinSF2.fillna(inplace=True, value=0)
data.BsmtFullBath.fillna(inplace=True, value=0)
data.BsmtHalfBath.fillna(inplace=True, value=0)
data.BsmtUnfSF.fillna(inplace=True, value=0)
data.GarageArea.fillna(inplace=True, value=0)
data.GarageCars.fillna(inplace=True, value=0)
data.GarageYrBlt.fillna(inplace=True, value=0)
data.LotFrontage.fillna(inplace=True, value=0)
data.MasVnrArea.fillna(inplace=True, value=0)
data.TotalBsmtSF.fillna(inplace=True, value=0)

# %% [code]

# Categorical fields
data.KitchenQual = data.KitchenQual.mode()[0]
data.Functional = data.Functional.mode()[0]
data.Utilities = data.Utilities.mode()[0]
data.SaleType = data.SaleType.mode()[0]
data.Exterior1st = data.Exterior1st.mode()[0]
data.Exterior2nd = data.Exterior2nd.mode()[0]
#change
data.Electrical = df['Electrical'].mode()[0]
data.MSZoning = data.MSZoning.mode()[0]
#change
data.MasVnrType = df['MasVnrType'].mode()[0]

print("After we imputed the missing values, the status of the data set is: ")
print(data.columns[data.isnull().any()])

# %% [code]

# Mappings that the ordinal features will use
# not all have a No options
garage_map = {'Unf': '2', 'RFn': '4', 'Fin': '6', 'No': '0'}
landslope_map = { 'Sev': '2', 'Mod': '4', 'Gtl': '6'}
lotshape_map = {'IR3': '2', 'IR2': '4', 'IR1': '6', 'Reg': '8'}
quality_map = {'Po': '2', 'Fa': '4', 'TA': '6', 'Gd': '8', 'Ex': '10', 'No': '0'}
utilities_map = {'ELO': '2', 'NoSeWa': '4', 'NoSewr': '6', 'AllPub': '8'}

# apply mappings
data.BsmtCond = data.BsmtCond.map(quality_map)
data.BsmtQual = data.BsmtQual.map(quality_map)
data.ExterCond = data.ExterCond.map(quality_map)
data.ExterQual = data.ExterQual.map(quality_map)
data.FireplaceQu = data.FireplaceQu.map(quality_map)
data.GarageCond = data.GarageCond.map(quality_map)
data.GarageFinish = data.GarageFinish.map(garage_map)
data.GarageQual = data.GarageQual.map(quality_map)
data.HeatingQC = data.HeatingQC.map(quality_map)
data.KitchenQual = data.KitchenQual.map(quality_map)
data.LandSlope = data.LandSlope.map(landslope_map)
data.LotShape = data.LotShape.map(lotshape_map)
data.PoolQC = data.PoolQC.map(quality_map)
data.Utilities = data.Utilities.map(utilities_map)

# change the data frame type to ints
data.BsmtCond = data.BsmtCond.astype('int64')
data.BsmtQual = data.BsmtQual.astype('int64')
data.ExterCond = data.ExterCond.astype('int64')
data.ExterQual = data.ExterQual.astype('int64')
data.FireplaceQu = data.FireplaceQu.astype('int64')
data.GarageCond = data.GarageCond.astype('int64')
data.GarageFinish = data.GarageFinish.astype('int64')
data.GarageQual = data.GarageQual.astype('int64')
data.HeatingQC = data.HeatingQC.astype('int64')
data.KitchenQual = data.KitchenQual.astype('int64')
data.LandSlope = data.LandSlope.astype('int64')
data.LotShape = data.LotShape.astype('int64')
data.PoolQC = data.PoolQC.astype('int64')
data.Utilities = data.Utilities.astype('int64')

# %% [code]

# making these categorical

#change all these
data['MSSubClass'] = data['MSSubClass'].astype("str")
data['YrSold'] = data['YrSold'].astype("str")
data['MoSold'] = data['MoSold'].astype("str")

# %% [code]

# making these ordinal
data.OverallCond = data.OverallCond.astype("int64")
data.OverallQual = data.OverallQual.astype("int64")
#change
data['KitchenAbvGr'] = data['KitchenAbvGr'].astype("int64")

# %% [code]

# one hot encoding
training_data = pd.get_dummies(data)
print("New  shape after one-hot encoding:", np.shape(training_data))

# %% [code]

# training_data['SalePrice'] = np.log1p(training_data.SalePrice)
training_data['TotalSF'] = training_data['TotalBsmtSF'] + training_data['1stFlrSF'] + training_data['2ndFlrSF'] + training_data[
    'GarageArea']

# %% [code]

# Random Forests Classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import metrics

y_train = training_data['SalePrice'].values
x_train = training_data.drop('SalePrice', axis=1).values

classifier = RandomForestRegressor(n_estimators=500, criterion='mse',
                                              max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                              min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                              min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                                              oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False)

kf = KFold(5, random_state=7, shuffle=True)
true_y = []
pred_y = []
fold = 0

for train, test in kf.split(x_train):
    fold += 1
    pred = []

    x_fold = x_train[train]
    y_fold = y_train[train]
    test_x_fold = x_train[test]
    test_y_fold = y_train[test]

    classifier.fit(x_fold, y_fold)
    pred = classifier.predict(test_x_fold)
    true_y.append(test_y_fold)
    pred_y.append(pred)

true_y = np.concatenate(true_y)
pred_y = np.concatenate(pred_y)
score = np.sqrt(metrics.mean_squared_error(true_y, pred_y))
print("\n Root Mean Square Error: {}".format(score))


# %% [code]
print(true_y)

# %% [code]
print(pred_y)