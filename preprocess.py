
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.feature_selection import VarianceThreshold

X_t = pd.read_csv('task1/X_train.csv', ',')
y_t = pd.read_csv('task1/y_train.csv', ',')


filler = SimpleImputer(missing_values=np.nan, strategy='mean')
filler.fit(X_t, y_t)

X_t = filler.transform(X_t)
pd.DataFrame(X_t).to_csv('task1/results/filled.csv', ',', index=False)

outlier_detection = IsolationForest(behaviour='new')
outlier_detection.fit(X_t[0:, 1:], y_t)

outlier = outlier_detection.predict(X_t[0:, 1:])
pd.DataFrame(outlier).to_csv('task1/results/outlier-decisions.csv', ',', index=False)

X_t = filter(lambda x: (outlier[int(x[0])] > 0), X_t)
pd.DataFrame(X_t).to_csv('task1/results/X_inliners.csv', ',', index=False)

y_t = filter(lambda x: (outlier[int(x[0])] > 0), y_t.values)
pd.DataFrame(y_t).to_csv('task1/results/y_inliners.csv', ',', index=False)


X_t = pd.read_csv('task1/results/X_inliners.csv', ',')
y_t = pd.read_csv('task1/results/y_inliners.csv', ',')

X_t = X_t.drop('0', axis=1)

scaler = StandardScaler()
scaler.fit(X_t)

X_t = scaler.transform(X_t)

pd.DataFrame(X_t).to_csv('task1/results/X_normalized.csv', ',', index=False)