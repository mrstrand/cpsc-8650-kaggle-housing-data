# %% [code]

#The script version of the kaggle notebook
#It is easier to work this way then copy and paste it into kaggle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
# %matplotlib inline

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", na_values=['?', ''], delimiter=',',delim_whitespace=False)
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", na_values=['?', ''], delimiter=',',delim_whitespace=False)
data = df
data_test = test

# %% [code]
# save off the id column for later
train_id = df['Id']
test_id = test['Id']

# %% [code]
# get the index of the outliers in GrLivArea
# the paper reccomends removing these
get_outliers = data.sort_values(by="GrLivArea", ascending=False).head(2)
get_outliers

# %% [code]
# drop the two outliers
data.drop(523, inplace=True)
data.drop(1298, inplace=True)
#data.drop([523, 1298], inplace=True)
# reset the index now that we have dropped two
data.reset_index(inplace=True)

# %% [code]
data_test.reset_index(inplace=True)

# %% [code]

print("Shape of training set: ", df.shape)
print("Shape of testing set: ", test.shape)

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

# %% [code]
# now do it to test as well
data_test.Alley.fillna(inplace=True, value='No')
data_test.BsmtCond.fillna(inplace=True, value='No')
data_test.BsmtExposure.fillna(inplace=True, value='No')
data_test.BsmtFinType1.fillna(inplace=True, value='No')
data_test.BsmtFinType2.fillna(inplace=True, value='No')
data_test.BsmtQual.fillna(inplace=True, value='No')
data_test.Fence.fillna(inplace=True, value='No')
data_test.FireplaceQu.fillna(inplace=True, value='No')
data_test.GarageCond.fillna(inplace=True, value='No')
data_test.GarageFinish.fillna(inplace=True, value='No')
data_test.GarageQual.fillna(inplace=True, value='No')
data_test.GarageType.fillna(inplace=True, value='No')
data_test.MiscFeature.fillna(inplace=True, value='No')
data_test.PoolQC.fillna(inplace=True, value='No')

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
# once again for test
data_test.BsmtFinSF1.fillna(inplace=True, value=0)
data_test.BsmtFinSF2.fillna(inplace=True, value=0)
data_test.BsmtFullBath.fillna(inplace=True, value=0)
data_test.BsmtHalfBath.fillna(inplace=True, value=0)
data_test.BsmtUnfSF.fillna(inplace=True, value=0)
data_test.GarageArea.fillna(inplace=True, value=0)
data_test.GarageCars.fillna(inplace=True, value=0)
data_test.GarageYrBlt.fillna(inplace=True, value=0)
data_test.LotFrontage.fillna(inplace=True, value=0)
data_test.MasVnrArea.fillna(inplace=True, value=0)
data_test.TotalBsmtSF.fillna(inplace=True, value=0)

# %% [code]

# Categorical fields
data.KitchenQual = data.KitchenQual.mode()[0]
data.Functional = data.Functional.mode()[0]
data.Utilities = data.Utilities.mode()[0]
data.SaleType = data.SaleType.mode()[0]
data.Exterior1st = data.Exterior1st.mode()[0]
data.Exterior2nd = data.Exterior2nd.mode()[0]
# data.Electrical = data.Electrical.mode()[0]
data.Electrical = df['Electrical'].mode()[0]
data.MSZoning = data.MSZoning.mode()[0]
data.MasVnrType = df['MasVnrType'].mode()[0]

# %% [code]
# test
data_test.KitchenQual = data_test.KitchenQual.mode()[0]
data_test.Functional = data_test.Functional.mode()[0]
data_test.Utilities = data_test.Utilities.mode()[0]
data_test.SaleType = data_test.SaleType.mode()[0]
data_test.Exterior1st = data_test.Exterior1st.mode()[0]
data_test.Exterior2nd = data_test.Exterior2nd.mode()[0]
data_test.Electrical = test['Electrical'].mode()[0]
data_test.MSZoning = data_test.MSZoning.mode()[0]
data_test.MasVnrType = test['MasVnrType'].mode()[0]

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
# apply mappings
data_test.BsmtCond = data_test.BsmtCond.map(quality_map)
data_test.BsmtQual = data_test.BsmtQual.map(quality_map)
data_test.ExterCond = data_test.ExterCond.map(quality_map)
data_test.ExterQual = data_test.ExterQual.map(quality_map)
data_test.FireplaceQu = data_test.FireplaceQu.map(quality_map)
data_test.GarageCond = data_test.GarageCond.map(quality_map)
data_test.GarageFinish = data_test.GarageFinish.map(garage_map)
data_test.GarageQual = data_test.GarageQual.map(quality_map)
data_test.HeatingQC = data_test.HeatingQC.map(quality_map)
data_test.KitchenQual = data_test.KitchenQual.map(quality_map)
data_test.LandSlope = data_test.LandSlope.map(landslope_map)
data_test.LotShape = data_test.LotShape.map(lotshape_map)
data_test.PoolQC = data_test.PoolQC.map(quality_map)
data_test.Utilities = data_test.Utilities.map(utilities_map)

# change the data frame type to ints
data_test.BsmtCond = data_test.BsmtCond.astype('int64')
data_test.BsmtQual = data_test.BsmtQual.astype('int64')
data_test.ExterCond = data_test.ExterCond.astype('int64')
data_test.ExterQual = data_test.ExterQual.astype('int64')
data_test.FireplaceQu = data_test.FireplaceQu.astype('int64')
data_test.GarageCond = data_test.GarageCond.astype('int64')
data_test.GarageFinish = data_test.GarageFinish.astype('int64')
data_test.GarageQual = data_test.GarageQual.astype('int64')
data_test.HeatingQC = data_test.HeatingQC.astype('int64')
data_test.KitchenQual = data_test.KitchenQual.astype('int64')
data_test.LandSlope = data_test.LandSlope.astype('int64')
data_test.LotShape = data_test.LotShape.astype('int64')
data_test.PoolQC = data_test.PoolQC.astype('int64')
data_test.Utilities = data_test.Utilities.astype('int64')

# %% [code]

# making these categorical

data['MSSubClass'] = data['MSSubClass'].astype("str")
data['YrSold'] = data['YrSold'].astype("str")
data['MoSold'] = data['MoSold'].astype("str")

# %% [code]
# test
data_test['MSSubClass'] = data_test['MSSubClass'].astype("str")
data_test['YrSold'] = data_test['YrSold'].astype("str")
data_test['MoSold'] = data_test['MoSold'].astype("str")

# %% [code]

# making these ordinal
data.OverallCond = data.OverallCond.astype("int64")
data.OverallQual = data.OverallQual.astype("int64")
data['KitchenAbvGr'] = data['KitchenAbvGr'].astype("int64")

# %% [code]
# test
data_test.OverallCond = data_test.OverallCond.astype("int64")
data_test.OverallQual = data_test.OverallQual.astype("int64")
data_test['KitchenAbvGr'] = data_test['KitchenAbvGr'].astype("int64")

# %% [code]

print(data.columns)
print(data_test.columns)

# %% [code]
# potentially doing this will eliminate the creating of extra featuers causing an error when
# sending the test data into the trained model
data['KitchenQual'] = pd.factorize(data['KitchenQual'])[0]
data['Functional'] = pd.factorize(data['Functional'])[0]
data['Utilities'] = pd.factorize(data['Utilities'])[0]
data['SaleType'] = pd.factorize(data['SaleType'])[0]
data['Exterior1st'] = pd.factorize(data['Exterior1st'])[0]
data['Exterior2nd'] = pd.factorize(data['Exterior2nd'])[0]
data['Electrical'] = pd.factorize(data['Electrical'])[0]
data['MSZoning'] = pd.factorize(data['MSZoning'])[0]
data['MasVnrType'] = pd.factorize(data['MasVnrType'])[0]

data['Street'] = pd.factorize(data['Street'])[0]
data['Alley'] = pd.factorize(data['Alley'])[0]
data['LandContour'] = pd.factorize(data['LandContour'])[0]
data['LotConfig'] = pd.factorize(data['LotConfig'])[0]
data['Neighborhood'] = pd.factorize(data['Neighborhood'])[0]
data['Condition1'] = pd.factorize(data['Condition1'])[0]
data['Condition2'] = pd.factorize(data['Condition2'])[0]
data['BldgType'] = pd.factorize(data['BldgType'])[0]
data['HouseStyle'] = pd.factorize(data['HouseStyle'])[0]
data['RoofStyle'] = pd.factorize(data['RoofStyle'])[0]
data['RoofMatl'] = pd.factorize(data['RoofMatl'])[0]
data['Foundation'] = pd.factorize(data['Foundation'])[0]
data['BsmtExposure'] = pd.factorize(data['BsmtExposure'])[0]
data['BsmtFinType1'] = pd.factorize(data['BsmtFinType1'])[0]
data['BsmtFinType2'] = pd.factorize(data['BsmtFinType2'])[0]
data['Heating'] = pd.factorize(data['Heating'])[0]
data['CentralAir'] = pd.factorize(data['CentralAir'])[0]
data['GarageType'] = pd.factorize(data['GarageType'])[0]
data['PavedDrive'] = pd.factorize(data['PavedDrive'])[0]

data['BsmtCond'] = pd.factorize(data['BsmtCond'])[0]
data['BsmtQual'] = pd.factorize(data['BsmtQual'])[0]
data['Fence'] = pd.factorize(data['Fence'])[0]
data['FireplaceQu'] = pd.factorize(data['FireplaceQu'])[0]
data['GarageCond'] = pd.factorize(data['GarageCond'])[0]
data['GarageFinish'] = pd.factorize(data['GarageFinish'])[0]
data['GarageType'] = pd.factorize(data['GarageType'])[0]
data['MiscFeature'] = pd.factorize(data['MiscFeature'])[0]
data['PoolQC'] = pd.factorize(data['PoolQC'])[0]

data['SaleCondition'] = pd.factorize(data['SaleCondition'])[0]

# %% [code]
data_test['KitchenQual'] = pd.factorize(data_test['KitchenQual'])[0]
data_test['Functional'] = pd.factorize(data_test['Functional'])[0]
data_test['Utilities'] = pd.factorize(data_test['Utilities'])[0]
data_test['SaleType'] = pd.factorize(data_test['SaleType'])[0]
data_test['Exterior1st'] = pd.factorize(data_test['Exterior1st'])[0]
data_test['Exterior2nd'] = pd.factorize(data_test['Exterior2nd'])[0]
data_test['Electrical'] = pd.factorize(data_test['Electrical'])[0]
data_test['MSZoning'] = pd.factorize(data_test['MSZoning'])[0]
data_test['MasVnrType'] = pd.factorize(data_test['MasVnrType'])[0]

data_test['Street'] = pd.factorize(data_test['Street'])[0]
data_test['Alley'] = pd.factorize(data_test['Alley'])[0]
data_test['LandContour'] = pd.factorize(data_test['LandContour'])[0]
data_test['LotConfig'] = pd.factorize(data_test['LotConfig'])[0]
data_test['Neighborhood'] = pd.factorize(data_test['Neighborhood'])[0]
data_test['Condition1'] = pd.factorize(data_test['Condition1'])[0]
data_test['Condition2'] = pd.factorize(data_test['Condition2'])[0]
data_test['BldgType'] = pd.factorize(data_test['BldgType'])[0]
data_test['HouseStyle'] = pd.factorize(data_test['HouseStyle'])[0]
data_test['RoofStyle'] = pd.factorize(data_test['RoofStyle'])[0]
data_test['RoofMatl'] = pd.factorize(data_test['RoofMatl'])[0]
data_test['Foundation'] = pd.factorize(data_test['Foundation'])[0]
data_test['BsmtExposure'] = pd.factorize(data_test['BsmtExposure'])[0]
data_test['BsmtFinType1'] = pd.factorize(data_test['BsmtFinType1'])[0]
data_test['BsmtFinType2'] = pd.factorize(data_test['BsmtFinType2'])[0]
data_test['Heating'] = pd.factorize(data_test['Heating'])[0]
data_test['CentralAir'] = pd.factorize(data_test['CentralAir'])[0]
data_test['GarageType'] = pd.factorize(data_test['GarageType'])[0]
data_test['PavedDrive'] = pd.factorize(data_test['PavedDrive'])[0]

data_test['BsmtCond'] = pd.factorize(data_test['BsmtCond'])[0]
data_test['BsmtQual'] = pd.factorize(data_test['BsmtQual'])[0]
data_test['Fence'] = pd.factorize(data_test['Fence'])[0]
data_test['FireplaceQu'] = pd.factorize(data_test['FireplaceQu'])[0]
data_test['GarageCond'] = pd.factorize(data_test['GarageCond'])[0]
data_test['GarageFinish'] = pd.factorize(data_test['GarageFinish'])[0]
data_test['GarageType'] = pd.factorize(data_test['GarageType'])[0]
data_test['MiscFeature'] = pd.factorize(data_test['MiscFeature'])[0]
data_test['PoolQC'] = pd.factorize(data_test['PoolQC'])[0]

data_test['SaleCondition'] = pd.factorize(data_test['SaleCondition'])[0]

# %% [code]

# one hot encoding
# training_data = pd.get_dummies(data)
# testing_data = pd.get_dummies(data_test)
# training_data = pd.factorize(data)
# testing_data = pd.factorize(data_test)
training_data = data
testing_data = data_test
# print("New  shape after one-hot encoding:", np.shape(training_data))

# %% [code]
training_data['TotalSF'] = training_data['TotalBsmtSF'] + training_data['1stFlrSF'] + training_data['2ndFlrSF'] + training_data['GarageArea']
testing_data['TotalSF'] = testing_data['TotalBsmtSF'] + testing_data['1stFlrSF'] + testing_data['2ndFlrSF'] + testing_data['GarageArea']

# %% [code]
print(training_data.shape)
print(testing_data.shape)

# %% [code]
# Random Forests Classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn import metrics

y_train = training_data['SalePrice'].values
x_train = training_data.drop('SalePrice', axis=1).values


classifier = RandomForestRegressor(n_estimators=50)

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
# validate that the classifier works
print("\n Root Mean Square Error: {}".format(score))

# %% [code]
print(np.shape(training_data))
print(np.shape(testing_data))

# %% [code]
print(len(x_train[0]))
print(x_train)

# %% [code]
test_set = testing_data.values
#test_set = training_data.values
print(len(test_set[0]))
print(test_set)

# %% [code]
# ValueError: Number of features of the model must match the input. Model n_features is 82 and input n_features is 83
# for some reason there is a feature being created in test that isnt in train
#test_set = training_data.values
classifier.fit(x_train,y_train)
prediction = classifier.predict(test_set)

# %% [code]
training_id = training_data['Id']
testing_id = testing_data['Id']
print(len(training_id))
print(len(testing_id))
print(len(pred_y))

# %% [code]
submit_test = pd.DataFrame()
submit_test['Id'] = testing_id
submit_test['SalePrice'] = prediction

# %% [code]
submit_train = pd.DataFrame()
submit_train['Id'] = training_id
submit_train['SalePrice'] = pred_y

# %% [code]
submit_test.to_csv('mySubmission.csv', index=False)
submit_train.to_csv('mySubmissionTrain.csv', index=False)