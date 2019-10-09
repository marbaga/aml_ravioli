

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

X_t = pd.read_csv('task1/X_train', ',')
y_t = pd.read_csv('task1/y_train', ',')

pipe = Pipeline([
                 ('scaler', StandardScaler()),
                 ('reduce_dim', PCA()),
                 ('regressor', Ridge())
                 ])


pipe = pipe.fit(X_t, y_t)
