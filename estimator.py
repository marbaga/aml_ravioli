
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

import warnings

from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=DeprecationWarning)

class InputTransform(BaseEstimator):

    def __init__(self, model, contamination='auto'):
        self.model = model
        self.contamination = contamination

        self.important_features = pd.read_csv('task1/best_features.csv', ',').to_numpy().flatten()
        self.useless_features = pd.read_csv('task1/useless_features.csv', ',').to_numpy().flatten()

        self.included_columns = []
        self.features_to_check = []


    def fill_data(self, X, y):
        self.filler = SimpleImputer(missing_values=np.nan, strategy='median')
        self.filler.fit(X, y)

        X = self.filler.transform(X)

        return (X, y)

    def detect_outliers(self, X, y):
        n = len([x for x in X]) - 1

        detector = LocalOutlierFactor(n_neighbors=max(50, int(0.1 * n)), contamination=self.contamination)
        decisions = detector.fit_predict(X, y)

        print('Outlier detection detected ' + str(len([x for x in decisions if x < 0])) + ' outliers.')

        L = lambda sample_id: (decisions[sample_id] > 0)

        X = [X[i] for i in range(0, len(decisions) - 1) if decisions[i] > 0]

        y = y.values
        y = [y[i] for i in range(0, len(decisions) - 1) if decisions[i] > 0]

        return (X, y)

    def select_features(self, X, y):
        X = pd.DataFrame(X)
        y = np.asarray(y).ravel()

        # Keep best features (re-merge later)
        X_best = X.copy().filter(self.important_features)

        self.features_to_check = []
        for i in range(0, 831):
            if i not in self.important_features and i not in self.useless_features:
                self.features_to_check.append(i)

        self.features_to_check = np.array(self.features_to_check)
        print('Features to check: ' + str(self.features_to_check.size))

        X = X.copy().filter(self.features_to_check)

        # correlation with output
        X.insert(X.shape[1], 'y', y, True)  # insert y

        cor = pd.DataFrame(X).corr()
        columns = np.full((cor.shape[0],), True, dtype=bool)
        for i in range(cor.shape[0]):
            for j in range(i + 1, cor.shape[0]):
                if abs(cor.iloc[i, j]) > 0.85 or abs(cor.iloc[j, cor.shape[0] - 1]) < 0.03:
                    if columns[j]:
                        columns[j] = False

        columns = columns[:-1].copy()

        del X['y']

        X = X.iloc[:, columns]
        #X = X.iloc[:, :X.shape[1] - 1]  # remove y

        self.included_columns = columns

        X = pd.concat([X_best.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

        print('Feature selection selected', X.shape[1], 'features.')

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = pd.DataFrame(self.scaler.transform(X))

        return (X, y)

    def fit(self, X, y, **fit_params):
        X, y = self.fill_data(X, y)
        X, y = self.detect_outliers(X, y)
        X, y = self.select_features(X, y)

        return self.model.fit(X, y)

    def predict(self, X):
        # fill with training median
        X = self.filler.transform(X)

        X = pd.DataFrame(X)
        # copy best features
        X_best = X.copy().filter(self.important_features)

        # filter medium features
        X = X.copy().filter(self.features_to_check)
        X = X.iloc[:, self.included_columns]

        # remerge best features
        X = pd.concat([X_best.reset_index(drop=True), X.reset_index(drop=True)], axis=1)

        X = self.scaler.transform(X)

        return self.model.predict(X)
