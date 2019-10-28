#CLASS TO

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBRegressor

# Contains a function to set parameters of the RandomForestRegressor with cv.
# Returns a model (or if you want a score).
# Uncomment and change code to make it useful again


def xbr_model(X,y):
    # Perform Grid-Search

    gsc = GridSearchCV(
        estimator=XGBRegressor(),
        param_grid={
            'learning_rate': (0.1, 0.3, 0.5),
            'max_depth': range(3, 7),
            'subsample': (0.3, 0.5, 0.7),
            'colsample_bytree': (0.1, 0.2, 0.4),
            'n_estimators': (10, 50, 100),
            'alpha': (0.1, 0.3, 0.5),
        },
        cv=5, scoring='r2', verbose=0)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    print('Best parameters: ')
    print(best_params)
    '''
    print(best_params["max_depth"])
    print(best_params["n_estimators"])'''

    rfr = XGBRegressor(learning_rate=best_params['learning_rate'],
                       max_depth=best_params['max_depth'], subsample=best_params['subsample'],
                       n_estimators=best_params['n_estimators'], colsample_bytree=best_params['colsample_bytree'],
                        alpha=best_params['alpha'])  # Perform K-Fold CV

    #scores = cross_val_score(rfr, X, y, cv=10, scoring='r2')

    return rfr