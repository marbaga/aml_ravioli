import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
from sklearn.model_selection import cross_validate, KFold, RepeatedKFold, LeaveOneOut, ShuffleSplit
from sklearn.preprocessing import StandardScaler
import warnings

X_t = pd.read_csv('task1/X_train.csv', ',').iloc[:, 1:]
y_t = pd.read_csv('task1/y_train.csv', ',').iloc[:, 1]

# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_t = filler.fit_transform(X_t, y_t)
X_t = pd.DataFrame(X_t)
print("Filled data with median")

# use lof
method = LocalOutlierFactor(n_neighbors=max(50, int(0.1*X_t.shape[1])), contamination=0.19)
outlier = method.fit_predict(X_t)
target = []
for i in range(0, len(outlier)):
    if outlier[i] == -1:
        target.append(i)
inliers = X_t.copy().drop(X_t.index[target], axis=0)
inliers_y = y_t.copy().drop(y_t.index[target])
print("Performed outlier detection")

all = pd.read_csv('task1/results/best_features.csv', ',').to_numpy().flatten()

selected_X = inliers.copy().filter(all)
selected_y =inliers_y
print("Performed feature selection. " + str(selected_X.shape) + ' is the final shape for the training matrix.')

scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(selected_X))
print("Standardized samples")

warnings.simplefilter(action='ignore', category=DeprecationWarning)

'''
#alpha found by RidgeCV
models = [linear_model.Ridge(tol=0.001, alpha=48)]
for m in models:
    cv_results = cross_validate(m, scaled_X, selected_y, cv=RepeatedKFold(n_repeats=100, n_splits=10), n_jobs=4)
    print('Score of ' + str(m) + ' (baseline)')
    print("Average: " + str(np.average(cv_results['test_score'])))
    print("Variance: " + str(np.var(cv_results['test_score'])))
    print()
'''

print('Preparing for prediction')

X_test = pd.read_csv('task1/X_test.csv', ',').iloc[:, 1:]
#print(X_test)
# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_test = filler.fit_transform(X_test)
X_test = pd.DataFrame(X_test)
#print(X_test)
print("Filled test with median")
X_test = X_test.copy().filter(all)
#print(X_test)
print("Performed feature selection. " + str(selected_X.shape) + ' is the final shape for the test matrix.')
scaler = StandardScaler()
X_test = pd.DataFrame(scaler.fit_transform(X_test))
#print(X_test)
print("Standardized test samples")

print("Predicting...")

m = linear_model.Ridge(tol=0.001, alpha=48)
m.fit(scaled_X, selected_y)
#print(m.score(scaled_X, selected_y))
predictions = m.predict(X_test)

print("Computing predictions")

answer = pd.read_csv('task1/X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=predictions, columns = ['y'])], axis = 1)
pd.DataFrame(answer).to_csv('task1/results/to_submit.csv', ',', index=False)

print("Prediction formattes and written to file")