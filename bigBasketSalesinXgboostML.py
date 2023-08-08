# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 07:46:28 2023

@author: 91955
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot
as plt
df =
pd.read_csv(r"C:\Users\91955\AppData\Local\Temp\Rar$DIa8364.41517/BigBasket Products.csv")
df['rating'].fillna(df['rating'].mean(), inplace=True)
df =
df.drop(['product', 'brand', 'type'], axis=1)
df2 = pd.get_dummies(df,
columns=["category", "sub_category"], drop_first=True)
X =
df2.drop("sale_price", axis=1)
y = df2["sale_price"]
from
sklearn.model_selection import train_test_split,
X_train, X_test, y_train, y_test =
train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import
LinearRegression
lr = LinearRegression()
xgb = XGBRegressor()
param_dist = {
 
'n_estimators': range(100, 1000, 100),
 'max_depth': range(3, 10),
 'learning_rate':
[0.01, 0.1, 0.2, 0.3],
 'subsample': [0.6, 0.7, 0.8, 0.9],
 'colsample_bytree': [0.6,
0.7, 0.8, 0.9],
 'gamma': [0, 1, 5],
 'reg_alpha': [0, 0.1, 0.5, 1],
 'reg_lambda':
[0, 0.1, 0.5, 1]
}
from sklearn.metrics import mean_squared_error,
mean_absolute_error
lr_reg = RandomizedSearchCV(estimator=lr, param_distributions=param_dist,
scoring="neg_root_mean_squared_error")
lr_reg.fit(X_train, y_train)
lr_y_pred =
lr_reg.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))
lr_mae =
mean_absolute_error(y_test, lr_y_pred)
print("Linear Regression RMSE:",
lr_rmse)
print("Linear Regression MAE:", lr_mae)
from xgboost import
XGBRegressor
xgb = XGBRegressor()
xgb_reg = RandomizedSearchCV(estimator=xgb,
param_distributions=param_dist,
scoring="neg_root_mean_squared_error")
xgb_reg.fit(X_train, y_train)
xgb_y_pred =
xgb_reg.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_y_pred))
xgb_mae =
mean_absolute_error(y_test, xgb_y_pred)
print("XGBoost RMSE:",
xgb_rmse)
print("XGBoost MAE:", xgb_mae)