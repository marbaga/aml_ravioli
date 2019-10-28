#CLASS TO

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel

# Contains a function to set parameters of the RandomForestRegressor with cv.
# Returns a model (or if you want a score).
# Uncomment and change code to make it useful again

def rfr_model(X,y):
    # Perform Grid-Search
    '''
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3, 7),
            'n_estimators': (10, 50, 100),
        },
        cv=5, scoring='r2', verbose=0)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    print('Best parameters: ')
    print(best_params["max_depth"])
    print(best_params["n_estimators"])
    '''
    rfr = RandomForestRegressor(max_depth=6, n_estimators=50,
                                random_state=False, verbose=False)  # Perform K-Fold CV

    #scores = cross_val_score(rfr, X, y, cv=10, scoring='r2')

    return rfr
