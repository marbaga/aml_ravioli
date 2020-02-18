import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from scipy.stats import shapiro, boxcox
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import ensemble
from sklearn.base import BaseEstimator

#Experiment with Gradient Boosting and RFE

class CustomEstimator (BaseEstimator):

    def __init__(self,
                 model=GradientBoostingRegressor(),
                 fill_nan=True,
                 fill_strategy='median',
                 contamination=0.015,
                 feature_selection='RFE',
                 features_to_select=180,
                 scale=True,
                 round_prediction_margin=0.2,
                 bootstrap=True):
        self.relevant_features = []
        self.not_uniform_features = []
        self.not_constant_features = []
        self.correlated_features = []
        self.RFE_features = []
        self.model = model
        self.fill_nan = fill_nan
        self.fill_strategy = fill_strategy
        self.contamination = contamination
        self.feature_selection = feature_selection
        self.features_to_select=features_to_select
        self.scale = scale
        self.round_prediction_margin = round_prediction_margin
        self.bootstrap = bootstrap
        self.scaler = StandardScaler()
        self.filler = SimpleImputer(missing_values=np.nan, strategy=self.fill_strategy)


    def fit(self, X, y):

        X_t = X.copy()
        y_t = y.copy()

        print('Filling Nans')
        if self.fill_nan :
            X_t = self.filler.fit_transform(X_t)

        print('Removing outliers')
        if self.contamination > 0:
            method = LocalOutlierFactor(n_neighbors=max(50, int(0.1 * X_t.shape[1])), contamination=self.contamination)
            outlier = method.fit_predict(X_t)
            indices = np.where(outlier == 1)
            X_t = X_t[indices, :][0, :, :]
            y_t = y_t[indices]

        if self.feature_selection == 'RFE':
            print('Removing features with zero variance')
            sel = VarianceThreshold()
            sel.fit(X_t)
            self.not_constant_features.extend(sel.get_support(indices=True))
            X_t = X_t[:, self.not_constant_features]

            print('Removing uniform features')
            if self.bootstrap:
                self.not_uniform_features.extend(pd.read_csv('task1/results/not_uniform.csv', ',').to_numpy().flatten())
            else:
                result = self.find_uniform_features(X_t, y_t)
                pd.DataFrame(result).to_csv('task1/results/not_uniform.csv', ',', index=False)
                self.not_uniform_features.extend(result)
            X_t = X_t[:, self.not_uniform_features]

            print('Removing highly correlated features')
            self.correlated_features.extend(self.find_correlated_features(0.9, 0.03, X_t, y_t))
            X_t = X_t[:, self.correlated_features]

            print('Running RFE')
            selector = RFE(estimator=self.model, n_features_to_select=self.features_to_select, step=100)
            selector = selector.fit(X_t, y_t)
            support = selector.get_support(indices=True)
            self.RFE_features.extend(support)
            X_t = X_t[:, self.RFE_features]

        print('Final training matrix shape is ' + str(X_t.shape))

        print('Scaling matrix')
        if self.scale:
            X_t = self.scaler.fit_transform(X_t)

        print('Fitting inner model')
        self.model.fit(X_t, y_t)
        print('Finished fitting')
        print()

        return self

    def predict(self, X):

        X_t = X.copy()

        if self.fill_nan :
            X_t = self.filler.transform(X_t)

        if self.feature_selection == 'RFE':
            X_t = X_t[:, self.not_constant_features]
            X_t = X_t[:, self.not_uniform_features]
            X_t = X_t[:, self.correlated_features]
            X_t = X_t[:, self.RFE_features]

        if self.scale:
            X_t = self.scaler.transform(X_t)

        predictions = self.model.predict(X_t)

        if self.round_prediction_margin>0:
            for i in range(0, len(predictions)):
                if np.abs(predictions[i] - int(round(predictions[i]))) < self.round_prediction_margin:
                    predictions[i] = int(round(predictions[i]))

        return predictions

    def find_uniform_features(self, X_t, y):
        ind = []
        for i in range(0, X_t.shape[1]):

            n, bins, a = plt.hist(X_t[:, i])
            #if n[1] + n[2] + n[9]+n[8] < n[5] +n[6]:
            if (2 * n[9] < n[5]):

                #plt.clf()
                #plt.show()
                ind.append(i)
            plt.clf()
        return ind

    def find_correlated_features(self, max_corr, min_corr, X, y):

        X2 = pd.DataFrame(X.copy())

        # insert y column
        X2.insert(X2.shape[1], 'y', y.copy(), True)

        # feature selection
        cor = pd.DataFrame(X2).corr()
        columns = np.full((cor.shape[0],), True, dtype=bool)
        for i in range(cor.shape[0]):
            if columns[i]:
                for j in range(i + 1, cor.shape[0]):
                    if columns[j]:
                        if abs(cor.iloc[i, j]) > max_corr:
                            columns[j] = False
                        if abs(cor.iloc[j, cor.shape[0] - 1]) < min_corr:
                            columns[j] = False

        to_keep = []
        for i in range(cor.shape[0]-1):
            if columns[i]:
                to_keep.append(i)

        return to_keep

X_t = pd.read_csv('task1/X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('task1/y_train.csv', ',').iloc[:, 1].to_numpy()
X_test = pd.read_csv('task1/X_test.csv', ',').iloc[:, 1:].to_numpy()

inner_model = GradientBoostingRegressor(learning_rate=0.01, max_depth=5, n_estimators=2500, subsample=0.7, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',random_state=10)
model = CustomEstimator(model=inner_model, fill_nan=True, fill_strategy='median', contamination=0.1, feature_selection='RFE', features_to_select=180, scale=True, round_prediction_margin=0.2, bootstrap=True)


cv_results = cross_validate(model, X_t, y_t.ravel(), cv=5, scoring='r2', n_jobs=-1,)

print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))

print('Predicting...')
model.fit(X_t, y_t.ravel())
predictions = model.predict(X_test)

print('Writing predictions')
answer = pd.read_csv('task1/X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=predictions, columns=['y'])], axis=1)
#pd.DataFrame(answer).to_csv('task1/results/gradient_boosting_regressor.csv', ',', index=False)

print("Prediction formatted and written to file")