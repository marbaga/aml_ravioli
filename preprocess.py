
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


filler = SimpleImputer(missing_values=np.nan, strategy='median')
filler.fit(X_t, y_t)

X_t = filler.transform(X_t)
pd.DataFrame(X_t).to_csv('task1/results/filled.csv', ',', index=False)


outlier_detections = [
    ('if', 'Isolation Forest', IsolationForest(behaviour='new', contamination='auto')),
    ('loc', 'Local Outlier Factor', LocalOutlierFactor(contamination='auto'))
]

decisions = {}

for (id, name, method) in outlier_detections:
    outlier = method.fit_predict(X_t[0:, 1:], y_t)

    previous_size = X_t.size
    pd.DataFrame(outlier).to_csv('task1/results/outlier-decisions.csv', ',', index=False)

    decisions[id] = outlier
    print(name + ' detected ' + str(len([x for x in outlier if x < 0])) + ' outliers.')


# decision function about outliers. true it inlier
L = lambda i: (decisions['loc'][i] > 0)


X_t = filter(lambda x: L(int(x[0])), X_t)
pd.DataFrame(X_t).to_csv('task1/results/X_inliners.csv', ',', index=False)

y_t = filter(lambda x: L(int(x[0])), y_t.values)
pd.DataFrame(y_t).to_csv('task1/results/y_inliners.csv', ',', index=False)

print(str(len([i for i in range(0, len(decisions['loc'])-1) if not L(i)])) + ' samples are considered outliers.')


X_t = pd.read_csv('task1/results/X_inliners.csv', ',')
y_t = pd.read_csv('task1/results/y_inliners.csv', ',')

X_t = X_t.drop('0', axis=1)

scaler = StandardScaler()
scaler.fit(X_t)

X_t = scaler.transform(X_t)

pd.DataFrame(X_t).to_csv('task1/results/X_normalized.csv', ',', index=False)