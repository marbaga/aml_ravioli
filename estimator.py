
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor


class InputTransform:

    def __init__(self, model):
        self.model = model

    def fill_data(self, X, y):
        filler = SimpleImputer(missing_values=np.nan, strategy='median')
        filler.fit(X, y)

        X = filler.transform(X)

        return (X, y)

    def detect_outliers(self, X, y):
        n = len([x for x in X]) - 1

        detector = LocalOutlierFactor(n_neighbors=max(50, int(0.1 * n)), contamination=0.03)
        decisions = detector.fit_predict(X[0:, 1:], y)

        print('Outlier detection detected ' + str(len([x for x in decisions if x < 0])) + ' outliers.')
        print([i for i in range(0, len(decisions) - 1) if decisions[i] < 0])

        # decision function about outliers. true if inlier
        L = lambda sample_id: (decisions[sample_id] > 0)

        outliers_count = len([i for i in range(0, len(decisions) - 1) if not L(i)])
        print(str(outliers_count) + ' samples are considered outliers (~' + str(
            int((outliers_count * 100) / n)) + '% of the set).')

        return pd.DataFrame(filter(lambda x: L(int(x[0])), X)), pd.DataFrame(filter(lambda x: L(int(x[0])), y.values))

    def select_features(self, X, y):
        # TODO
        return (X, y)


    def fit(self, X, y):
        X, y = self.fill_data(X, y)
        X, y = self.detect_outliers(X, y)
        X, y = self.select_features(X, y)

        return self.model.fit(X, y)


