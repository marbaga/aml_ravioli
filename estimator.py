
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
from sklearn.model_selection import cross_validate, KFold, RepeatedKFold, LeaveOneOut, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

import warnings
import random


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
        print([i for i in range(0, len(decisions) - 1) if decisions[i] < 0])

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
        y = np.asarray(y)

        # insert y column
        X.insert(X.shape[1], 'y', y, True)

        # eliminate features with variance 0
        sel = VarianceThreshold(threshold=0)
        X = sel.fit_transform(X)

        # feature selection
        cor = pd.DataFrame(X).corr()
        columns = np.full((cor.shape[0],), True, dtype=bool)
        for i in range(cor.shape[0]):
            for j in range(i + 1, cor.shape[0]):
                if abs(cor.iloc[i, j]) > 0.95 or abs(cor.iloc[j, cor.shape[0] - 1]) < 0.05:
                    if columns[j]:
                        columns[j] = False
        selected_columns = pd.DataFrame(X).columns[columns]
        data = pd.DataFrame(X[:, selected_columns])

        # removing y
        data = data.iloc[:, :data.shape[1] - 1]

        # first small selection (creates 'all' vector), probably not the best way
        ridge = Ridge(alpha=1.0)
        selector = RFE(estimator=ridge, n_features_to_select=20, step=10)
        selector = selector.fit(data, y.ravel())
        all = selector.get_support(indices=True).flatten()

        # greedy forward feature selection
        added_features = []
        newset = []
        found = True
        while len(all) < 200 or found:
            found = False
            xx = data.copy().filter(all)
            yy = y
            m = linear_model.Ridge()
            scaler = StandardScaler()
            xx = pd.DataFrame(scaler.fit_transform(xx))
            cv_results = cross_validate(m, xx, yy, cv=RepeatedKFold(n_repeats=50, n_splits=10), n_jobs=-1)
            print('Score of ' + str(m) + ' (baseline)')
            print("Average: " + str(np.average(cv_results['test_score'])))

            baseline = np.average(cv_results['test_score'])
            lis = list(range(830))
            random.shuffle(lis)
            for i in lis:
                newset = np.append(all.copy(), i)
                xx = data.copy().filter(newset)
                xx = pd.DataFrame(scaler.fit_transform(xx))
                cv_results = cross_validate(m, xx, yy, cv=RepeatedKFold(n_repeats=50, n_splits=10), n_jobs=-1)
                print('Score of ' + str(m) + 'adding feature ' + str(i))
                print("Average: " + str(np.average(cv_results['test_score'])))
                score = np.average(cv_results['test_score'])
                if score > baseline * 1.002:
                    all = np.append(all, i)
                    added_features = np.append(added_features, i)
                    found = True
                    break
                print("Checked feature " + str(i) + ". Added features: " + str(added_features))
        X = data.copy().filter(newset)
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
        return self.model.predict(X)
