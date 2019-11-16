import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import imblearn
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import VotingClassifier
from keras import regularizers, optimizers

class CustomEstimator (BaseEstimator):

    def __init__(self, sampler,
                 model
                 ):
        self.model = model
        self.indices = []
        self.sampler = sampler
        self.scaler = StandardScaler()

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
        #print('Final training matrix shape is ' + str(X_t.shape))
        X_t = self.scaler.fit_transform(X_t)
        self.model.fit(X_t, y_t)
        return self

    def predict(self, X):

        X_t = X.copy()
        X_t = self.scaler.transform(X_t)
        predictions = self.model.predict(X_t)
        return predictions

    def predict_proba(self, X):
        X_t = X.copy()
        X_t = self.scaler.transform(X_t)
        predictions = self.model.predict_proba(X_t)
        return predictions

class CustomNN(BaseEstimator):

    def __init__(self, model, sampler=imblearn.under_sampling.RandomUnderSampler(), sample=False, weight=False, class_weight=None):
        self.model = model
        self.sampler = sampler
        self.sample=sample
        self.weight=weight
        self.class_weight = class_weight
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_t = X.copy()
        y_t = y.copy()
        if self.sample:
            X_t, y_t = self.sampler.fit_sample(X_t, y_t)
        X_t = self.scaler.fit_transform(X_t)
        if self.weight:
            self.model.fit(X_t, y_t, class_weight=self.class_weight, verbose=0)
        else:
            self.model.fit(X_t, y_t, verbose=0)
        return self

    def predict(self, X):
        X_t = X.copy()
        X_t = self.scaler.transform(X_t)
        predictions = self.model.predict(X_t)
        return predictions

    def predict_proba(self, X):
        X_t = X.copy()
        X_t = self.scaler.transform(X_t)
        predictions = self.model.predict_proba(X_t)
        return predictions

def baseline_model_weight():
    model = Sequential()
    model.add(Dense(700, input_dim=1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002))
    return model

def baseline_model_sample():
    model = Sequential()
    model.add(Dense(700, input_dim=1000, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0002))
    return model


#This bit performs cross validation. Every transformation on data needs to be carried out inside the custom estimator class. The following lines should not be touched
X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()

###################################################################################

###################################################################################

voters=[]
for i in range(0,3):
    voters.append(('weighted_nn' + str(i), CustomNN(model=KerasClassifier(build_fn=baseline_model_weight, epochs=10, batch_size=32), weight=True, class_weight={0:1, 1:0.16, 2:1})))
    voters.append(('sampled_nn'+str(i), CustomNN(model=KerasClassifier(build_fn=baseline_model_sample, epochs=8, batch_size=32), sample=True)))
    voters.append(('gradboost'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=xgb.XGBClassifier(ax_depth=3, n_estimators=600, learning_rate=0.16, subsample=0.5, colsample_bytree=0.5, verbose=False))))
    voters.append(('svc'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=SVC(kernel='rbf', gamma='scale', shrinking=False))))
    voters.append(('svc2'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=SVC(kernel='rbf', gamma='scale', shrinking=False))))

    #voters.append(('logreg'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=LogisticRegression(solver='saga', penalty='l1', C=0.5, max_iter=200, multi_class='auto'))))

'''
for model in voters:
    cv_results = cross_validate(model[1], X_t, y_t, scoring='balanced_accuracy', n_jobs=-1, cv=10, verbose=True)
    print('Score of ' + str(model) + ': ')
    print(cv_results['test_score'])
    print("Average: " + str(np.average(cv_results['test_score'])))
    print("Variance: " + str(np.var(cv_results['test_score'])))
'''

model = VotingClassifier(voters)
cv_results = cross_validate(model, X_t, y_t, scoring='balanced_accuracy', cv=10, verbose=True)
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
print("Variance: " + str(np.var(cv_results['test_score'])))
'''
model.fit(X_t, y_t)
pred = model.predict(X_test)
answer = pd.read_csv('X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=pred, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('result3.csv', ',', index=False)'''

###################################################################################

voters=[]
for i in range(0,4):
    voters.append(('weighted_nn' + str(i), CustomNN(model=KerasClassifier(build_fn=baseline_model_weight, epochs=10, batch_size=32), weight=True, class_weight={0:1, 1:0.16, 2:1})))
    voters.append(('sampled_nn'+str(i), CustomNN(model=KerasClassifier(build_fn=baseline_model_sample, epochs=8, batch_size=32), sample=True)))
    voters.append(('gradboost'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=xgb.XGBClassifier(ax_depth=3, n_estimators=600, learning_rate=0.16, subsample=0.5, colsample_bytree=0.5, verbose=False))))
    voters.append(('svc'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=SVC(kernel='rbf', gamma='scale', shrinking=False))))

    #voters.append(('logreg'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=LogisticRegression(solver='saga', penalty='l1', C=0.5, max_iter=200, multi_class='auto'))))

'''
for model in voters:
    cv_results = cross_validate(model[1], X_t, y_t, scoring='balanced_accuracy', n_jobs=-1, cv=10, verbose=True)
    print('Score of ' + str(model) + ': ')
    print(cv_results['test_score'])
    print("Average: " + str(np.average(cv_results['test_score'])))
    print("Variance: " + str(np.var(cv_results['test_score'])))
'''

model = VotingClassifier(voters)
cv_results = cross_validate(model, X_t, y_t, scoring='balanced_accuracy', cv=10, verbose=True)
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
print("Variance: " + str(np.var(cv_results['test_score'])))
'''
model.fit(X_t, y_t)
pred = model.predict(X_test)
answer = pd.read_csv('X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=pred, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('result4.csv', ',', index=False)'''

###################################################################################

voters=[]
for i in range(0,4):
    voters.append(('weighted_nn' + str(i), CustomNN(model=KerasClassifier(build_fn=baseline_model_weight, epochs=10, batch_size=32), weight=True, class_weight={0:1, 1:0.16, 2:1})))
    voters.append(('sampled_nn'+str(i), CustomNN(model=KerasClassifier(build_fn=baseline_model_sample, epochs=8, batch_size=32), sample=True)))
    voters.append(('gradboost'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=xgb.XGBClassifier(ax_depth=3, n_estimators=600, learning_rate=0.16, subsample=0.5, colsample_bytree=0.5, verbose=False))))
    voters.append(('svc'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=SVC(kernel='rbf', gamma='scale', shrinking=False))))
    voters.append(('svc2'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=SVC(kernel='rbf', gamma='scale', shrinking=False))))

    #voters.append(('logreg'+str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=LogisticRegression(solver='saga', penalty='l1', C=0.5, max_iter=200, multi_class='auto'))))

'''
for model in voters:
    cv_results = cross_validate(model[1], X_t, y_t, scoring='balanced_accuracy', n_jobs=-1, cv=10, verbose=True)
    print('Score of ' + str(model) + ': ')
    print(cv_results['test_score'])
    print("Average: " + str(np.average(cv_results['test_score'])))
    print("Variance: " + str(np.var(cv_results['test_score'])))
'''

model = VotingClassifier(voters)
cv_results = cross_validate(model, X_t, y_t, scoring='balanced_accuracy', cv=10, verbose=True)
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))
print("Variance: " + str(np.var(cv_results['test_score'])))
'''
model.fit(X_t, y_t)
pred = model.predict(X_test)
answer = pd.read_csv('X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=pred, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('result5.csv', ',', index=False)'''
