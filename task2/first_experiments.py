'''
So far I experimented with different classifiers and differend samplers.
Those delivering better results seem to be logistic regression, NuSVC, GradientBoostingClassifier, XGBClassifier and MLPClassifier.
Random undersampling seems to obtain better results.
Best score is around 0.69 (SVC, RandomUnderSampler)
Shallow neural networks with dropout and hidden layer achieve similar performance.

Note that all models have a very long training time on the whole subset.
All features have variance > 0.
All features I have seen seem to follow a Gaussian-ish distribution.
By uncommenting the lines in method fit you can visualize boxplots for the 3 classes.

Ideas:
- eliminate features for which the boxplots of all classes are very similar
- try more fancy feature selection methods
- try deeper neural network (running on a GPU through Google Colab)
- stack best performing classifiers
- try again with other sampling techniques
- following Kaggle Kernels step by steps
- finally, tune hyperparameters


- perform outlier detection (with a lot of care) -> apparently not good
- perform feature selection basing of correlation between features -> PCA does not have major effects
'''

import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
import imblearn
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.neural_network import MLPClassifier
class CustomEstimator (BaseEstimator):

    def __init__(self, sampler,
                 model, early_stop=False
                 ):

        self.early_stop = early_stop
        self.model = model
        self.indices = []
        self.sampler = sampler
        self.scaler = StandardScaler()

    def fit(self, X, y):

        X_t = X.copy()
        y_t = y.copy()
        X_t, y_t = self.sampler.fit_sample(X_t, y_t)

        #self.feature_selector = PCA(n_components=500)
        #self.feature_selector.fit(X_t)
        #X_t = self.feature_selector.transform(X_t)

        '''
        for i in range(0, X_t.shape[1]):
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
            sns.boxplot(X_t[np.where(y_t == 0), i], color='blue', ax=ax1)
            sns.boxplot(X_t[np.where(y_t == 1), i], color='red', ax=ax2)
            sns.boxplot(X_t[np.where(y_t == 2), i], color='pink', ax=ax3)
            plt.show()
            '''
        print('Final training matrix shape is ' + str(X_t.shape))

        X_t = self.scaler.fit_transform(X_t)

        if self.early_stop:
            early_stopping = EarlyStopping(
                monitor='loss',
                verbose=1,
                patience=15,
                mode='min',
                baseline=0.04
            )
            self.model.fit(X_t, y_t, callbacks = [early_stopping])
        else:
            self.model.fit(X_t, y_t)

        return self

    def predict(self, X):

        X_t = X.copy()
        #X_t = self.feature_selector.transform(X_t)
        X_t = self.scaler.transform(X_t)
        predictions = self.model.predict(X_t)

        return predictions

def baseline_model():
    model = Sequential()
    model.add(Dense(700, input_dim=1000, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

def build_keras_classifier():
    return KerasClassifier(build_fn=baseline_model, epochs=32, batch_size=256)

def build_svc_classifier():
    return SVC(kernel='rbf', gamma='scale', shrinking=False)

#This bit performs cross validation. Every transformation on data needs to be carried out inside the custom estimator class. The following lines should not be touched
X_t = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('y_train.csv', ',').iloc[:, 1].to_numpy()
X_test = pd.read_csv('X_test.csv', ',').iloc[:, 1:].to_numpy()

voters = []
for i in range(1, 4):
    voters.append(('nn'+ str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=build_keras_classifier(), early_stop=True)))

for i in range(1, 4):
    voters.append(('svc'+ str(i), CustomEstimator(sampler=imblearn.under_sampling.RandomUnderSampler(), model=build_svc_classifier())))



model = VotingClassifier(voters)

cv_results = cross_validate(model, X_t, y_t, scoring='balanced_accuracy', n_jobs=-1, cv=10, verbose=True)
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))

#model.fit(X_t, y_t)
#pred = model.predict(X_test)
#answer = pd.read_csv('X_test.csv', ',')[['id']]
#answer = pd.concat([answer, pd.DataFrame(data=pred, columns=['y'])], axis=1)
#pd.DataFrame(answer).to_csv('result.csv', ',', index=False)


#Collection of decent ideas
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