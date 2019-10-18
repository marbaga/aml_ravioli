
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

import warnings


warnings.simplefilter(action='ignore', category=DeprecationWarning)

class InputTransform(BaseEstimator):

    def __init__(self, model, contamination='auto'):
        self.model = model
        self.contamination = contamination

    def fill_data(self, X, y):
        filler = SimpleImputer(missing_values=np.nan, strategy='median')
        filler.fit(X, y)

        X = filler.transform(X)

        return (X, y)

    def detect_outliers(self, X, y):
        n = len([x for x in X]) - 1

        detector = LocalOutlierFactor(n_neighbors=max(50, int(0.1 * n)), contamination=self.contamination)
        decisions = detector.fit_predict(X, y)

        print('Outlier detection detected ' + str(len([x for x in decisions if x < 0])) + ' outliers.')
        #print([i for i in range(0, len(decisions) - 1) if decisions[i] < 0])

        # decision function about outliers. true if inlier
        L = lambda sample_id: (decisions[sample_id] > 0)

        #outliers_count = len([i for i in range(0, len(decisions) - 1) if not L(i)])
        #print(str(outliers_count) + ' samples are considered outliers (~' + str(
        #    int((outliers_count * 100) / n)) + '% of the set).')

        X = [X[i] for i in range(0, len(decisions) - 1) if decisions[i] > 0]

        y = y.values
        y = [y[i] for i in range(0, len(decisions) - 1) if decisions[i] > 0]

        return (X, y)

    def select_features(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y).ravel()

        self.var_threshold = VarianceThreshold(threshold=0.1)
        X = self.var_threshold.fit_transform(X)

        clf = RandomForestClassifier(n_estimators=50)
        clf = clf.fit(X, y)

        self.feature_selector = SelectFromModel(clf, prefit=True, max_features=200)
        X = self.feature_selector.transform(X)

        print('Feature selection selected', X.shape[1], 'features.')

        return (X, y)


    def fit(self, X, y, **fit_params):
        X, y = self.fill_data(X, y)
        X, y = self.detect_outliers(X, y)
        X, y = self.select_features(X, y)

        return self.model.fit(X, y)

    def predict(self, X):
        filler = SimpleImputer(missing_values=np.nan, strategy='median')
        filler.fit(X)

        X = filler.transform(X)
        X = self.var_threshold.transform(X)
        X = self.feature_selector.transform(X)

        return self.model.predict(X)
