
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

n = len([x for x in X_t]) - 1

# changing n_neighbors to max(50, int(0.1*n)) considerably increases the number of outliers (default was 20)
outlier_detections = [
    #('if', 'Isolation Forest', IsolationForest(behaviour='new', contamination='auto')),
    ('loc', 'Local Outlier Factor', LocalOutlierFactor(n_neighbors=max(50, int(0.1*n)), contamination='auto'))
]

decisions = {}

for (id, name, method) in outlier_detections:
    outlier = method.fit_predict(X_t[0:, 1:], y_t)

    previous_size = X_t.size
    pd.DataFrame(outlier).to_csv('task1/results/outlier-decisions.csv', ',', index=False)

    decisions[id] = outlier
    print(name + ' detected ' + str(len([x for x in outlier if x < 0])) + ' outliers.')
    print([i for i in range(0, len(outlier) - 1) if outlier[i] < 0])


# decision function about outliers. true if inlier
L = lambda sample_id: (decisions['loc'][sample_id] > 0)


X_t = filter(lambda x: L(int(x[0])), X_t)
pd.DataFrame(X_t).to_csv('task1/results/X_inliners.csv', ',', index=False)

y_t = filter(lambda x: L(int(x[0])), y_t.values)
pd.DataFrame(y_t).to_csv('task1/results/y_inliners.csv', ',', index=False)

outliers_count = len([i for i in range(0, len(decisions['loc'])-1) if not L(i)])
print(str(outliers_count) + ' samples are considered outliers (~'+ str(int((outliers_count*100)/n)) +'% of the set).')


X_t = pd.read_csv('task1/results/X_inliners.csv', ',')
y_t = pd.read_csv('task1/results/y_inliners.csv', ',')

#X_t = X_t.drop('0', axis=1)
#
#scaler = StandardScaler()
#scaler.fit(X_t)
#
#X_t = scaler.transform(X_t)
#
#pd.DataFrame(X_t).to_csv('task1/results/X_normalized.csv', ',', index=False)