
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline

X_t = pd.read_csv('task1/X_train.csv', ',')
y_t = pd.read_csv('task1/y_train.csv', ',')

pipe = Pipeline([
                 ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                 ('outlier_detection', LocalOutlierFactor())
                 ])

pipe = pipe.fit(X_t, y_t)

X_t_filled = pipe.transform(X_t)

pd.DataFrame(X_t_filled[0:, 1:]).to_csv('task1/output.csv', ',')
