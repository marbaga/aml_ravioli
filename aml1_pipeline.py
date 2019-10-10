
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline


from sklearn.feature_selection import VarianceThreshold

X_t = pd.read_csv('task1/X_train.csv', ',')
y_t = pd.read_csv('task1/y_train.csv', ',')


filler = SimpleImputer(missing_values=np.nan, strategy='mean')
filler.fit(X_t, y_t)

X_t = filler.transform(X_t)
pd.DataFrame(X_t).to_csv('task1/results/filled.csv', ',')

outlier_detection = IsolationForest(behaviour='new')
outlier_detection.fit(X_t[0:, 1:], y_t)

outlier = outlier_detection.predict(X_t[0:, 1:])
pd.DataFrame(outlier).to_csv('task1/results/outlier-decisions.csv', ',')

X_t = filter(lambda x: (outlier[int(x[0])] > 0), X_t)
pd.DataFrame(X_t).to_csv('task1/results/X_inliners.csv', ',')

y_t = filter(lambda x: (outlier[int(x[0])] > 0), y_t.values)
pd.DataFrame(y_t).to_csv('task1/results/y_inliners.csv', ',')






