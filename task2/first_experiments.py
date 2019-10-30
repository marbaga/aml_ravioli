'''
So far I experimented with different classifiers and differend samplers.
Those delivering better results seem to be logistic regression, NuSVC, GradientBoostingClassifier, XGBClassifier and MLPClassifier.
Random undersampling seems to obtain better results.
Best score is around 0.68 (NuSVC, RandomUnderSampler

Note that all models have a very long training time on the whole subset.
Please check if cross-validation is performed correctly (all processes should be inside methods in CustomEstimator.
All features have variance > 0.
All features I have seen seem to follow a Gaussian-ish distribution.
By uncommenting the lines in method fit you can visualize boxplots for the 3 classes.

Ideas:
- perform outlier detection (with a lot of care)
- perform feature selection basing of correlation between features
- eliminate features for which the boxplots of all classes are very similar
- try more fancy feature selection methods
- try feature engineering if we are given more informations
- try deeper neural network (running on a GPU through Google Colab)
- stack best performing classifiers
- following Kaggle Kernels step by steps
- finally, tune hyperparameters

'''




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from scipy.stats import shapiro, boxcox
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import ensemble
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import imblearn
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


class CustomEstimator (BaseEstimator):

    def __init__(self, sampler,
                 model
                 ):
        self.model = model
        self.indices = []
        self.sampler = sampler

    def fit(self, X, y):

        X_t = X.copy()
        y_t = y.copy()
        X_t, y_t = self.sampler.fit_sample(X_t, y_t)

        '''
        for i in range(0, X_t.shape[1]):
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
            sns.boxplot(X_t[np.where(y_t == 0), i], color='blue', ax=ax1)
            sns.boxplot(X_t[np.where(y_t == 1), i], color='red', ax=ax2)
            sns.boxplot(X_t[np.where(y_t == 2), i], color='pink', ax=ax3)
            plt.show()
            '''

        self.feature_selector = PCA(n_components = 500)
        self.feature_selector.fit(X_t)
        X_t = self.feature_selector.transform(X_t)

        #self.scaler = StandardScaler()
        #self.scaler.fit(X_t)
        #X_t = self.scaler.transform(X_t)

        print('Final training matrix shape is ' + str(X_t.shape))
        self.model.fit(X_t, y_t)

        return self

    def predict(self, X):

        X_t = X.copy()

        X_t = self.feature_selector.transform(X_t)
        #X_t = self.scaler.transform(X_t)

        y_t = self.model.predict(X_t)

        return y_t

#This bit performs cross validation. Every transformation on data needs to be carried out inside the custom estimator class. The following lines should not be touched
X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()

classifiers = [
    LogisticRegression(),
    NuSVC(probability=True),
    GradientBoostingClassifier(),
    MLPClassifier(),
    XGBClassifier()]

samplers = [imblearn.under_sampling.TomekLinks(ratio='majority', n_jobs=-1),
            imblearn.combine.SMOTETomek(ratio='auto', n_jobs=-1),
            imblearn.under_sampling.ClusterCentroids(ratio={1:600}, n_jobs=-1),
            imblearn.over_sampling.RandomOverSampler(),
            imblearn.over_sampling.SMOTE(ratio='minority', n_jobs=-1),
            imblearn.under_sampling.RandomUnderSampler()]

for clf in classifiers:
    model = CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=clf)
    cv_results = cross_validate(model, X_t, y_t, scoring='balanced_accuracy', n_jobs=-1, cv=10, verbose=True)
    print('Score of ' + str(model) + ': ')
    print(cv_results['test_score'])
    print("Average: " + str(np.average(cv_results['test_score'])))