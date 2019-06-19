import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection  import  cross_val_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
#%matplotlib inline


train_data = pd.read_csv('./home-data/train.csv')
test_data = pd.read_csv('./home-data/test.csv')


train_data.dropna(axis=0, subset = ['SalePrice'], inplace = True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis = 1, inplace = True)



numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
low_factor_vars = [cname for cname in train_data.columns if train_data[cname].dtype == "object"]
cols_2_keep = numeric_cols + low_factor_vars



X = train_data[cols_2_keep].copy()
X_test = test_data[cols_2_keep].copy()


X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X, X_test = X.align(X_test, join='left', axis=1)



def get_mae_score(n_estimators=100, learning_rate=0.1, n_jobs = 2):
    xgb_model = XGBRegressor(n_estimators = n_estimators, learning_rate=learning_rate, n_jobs=n_jobs)
    mae_score = -1 * cross_val_score(xgb_model, X, y, cv = 10, verbose = 0, scoring='neg_mean_absolute_error')
    return print( 'Average MAE Score:', mae_score.mean())



results = {}
params = [700 + i*100 for i in range(0,4)]
for n in params:
    results[n] = get_mae_score(n_estimators=n, learning_rate=0.1)


plt.plot(results.keys(), results.values())
plt.show()



#def get_rmse_score(n_estimators=100, learning_rate=0.1, n_jobs = 2):
#    xgb_model = XGBRegressor(n_estimators = n_estimators, learning_rate=learning_rate, n_jobs=n_jobs)
#    rmse_score = -1 * cross_val_score(xgb_model, X, y, cv = 10, verbose = 0, scoring = 'neg_mean_squared_error')
#    return print( 'Average RMSE Score:', rmse_score.mean())


#get_rmse_score(n_estimators=600, learning_rate=0.1)









