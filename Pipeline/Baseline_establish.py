#establish a baseline score to judge against our engineered features

## function to compute cross-validated RMSLE score for a feature set
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import json

# fetch the data from data/raw
df_train = pd.read_csv('./data/raw/df_train.csv')
df_test = pd.read_csv('./data/raw/df_test.csv')


def score_dataset(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(['category']):
        X[colname] = X[colname].cat.codes
    
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error"
    )
    score = -1*score.mean()
    score = np.sqrt(score)
    return score

X = df_train.copy()
y = X.pop('SalePrice')


baseline_score = score_dataset(X, y)


metrics_dict= {"Baseline score": baseline_score}
with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)