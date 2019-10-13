import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
from sklearn.model_selection import cross_validate, KFold, RepeatedKFold, LeaveOneOut, ShuffleSplit
from sklearn.preprocessing import StandardScaler
import warnings
import random


#this script can be used for greedy forward feature selection
X_t = pd.read_csv('task1/X_train.csv', ',').iloc[:, 1:]
y_t = pd.read_csv('task1/y_train.csv', ',').iloc[:, 1]

# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_t = filler.fit_transform(X_t, y_t)
X_t = pd.DataFrame(X_t)

print("Filled data with median")

inliers = []
inliers_y = []

# use lof
method = LocalOutlierFactor(n_neighbors=max(50, int(0.1*X_t.shape[1])), contamination=0.19)
outlier = method.fit_predict(X_t)
target = []
for i in range(0, len(outlier)):
    if outlier[i] == -1:
        target.append(i)
inliers.append(X_t.copy().drop(X_t.index[target], axis=0))
inliers_y.append(y_t.copy().drop(y_t.index[target]))

print("Added datasets calculated with LOF")


selected = []
selectedy = []

all = []
for i in [8, 13, 52, 59, 75, 83, 93, 95]:
    new = pd.read_csv('task1/results/to_keep'+str(i)+'.csv', ',')
    all.extend(new.to_numpy().flatten())

set2 = set(all)
set2.add(0)
all = np.sort(list(set2))
print(len(all))

#features already selected by running this script
l = np.array([786, 386, 423, 96, 93, 6, 680, 82, 794, 304, 15, 94, 348, 490, 210, 254, 489, 319, 215, 406, 135, 24, 628, 4, 120, 246, 171, 624, 172, 149])
all = np.concatenate([all, l])
print(len(all))
all.sort()
pd.DataFrame(all).to_csv('task1/results/best_features.csv', ',', index=False)
selected.append(inliers[0].copy().filter(all))
selectedy.append(inliers_y[0])

warnings.simplefilter(action='ignore', category=DeprecationWarning)

added_features = []
found = True
while len(all)<200 or found:
    found = False
    xx = inliers[0].copy().filter(all)
    yy = inliers_y[0]
    m = linear_model.RidgeCV(alphas=[0.01, 0.05, 0.1, 0.5, 1, 5], cv=10)
    scaler = StandardScaler()
    xx = pd.DataFrame(scaler.fit_transform(xx))
    cv_results = cross_validate(m, xx, yy, cv=RepeatedKFold(n_repeats=100, n_splits=10), n_jobs=4)
    print('Score of ' + str(m) + ' (baseline)')
    print("Average: " + str(np.average(cv_results['test_score'])))

    baseline = np.average(cv_results['test_score'])
    lis = list(range(830))
    random.shuffle(lis)
    for i in lis:
        newset = np.append(all.copy(), i)
        xx = inliers[0].copy().filter(newset)
        xx = pd.DataFrame(scaler.fit_transform(xx))
        cv_results = cross_validate(m, xx, yy, cv=RepeatedKFold(n_repeats=50, n_splits=10), n_jobs=4)
        print('Score of ' + str(m) + 'adding feature ' + str(i))
        print("Average: " + str(np.average(cv_results['test_score'])))
        score = np.average(cv_results['test_score'])
        if score > baseline*1.002:
            all = np.append(all, i)
            added_features = np.append(added_features, i)
            found = True
            break
        print("Checked feature " + str(i) + ". Added features: " + str(added_features))


#these last lines were used to compare the inclusion of 1 feature to a predetermined set.
'''
count = 0
max = 0
maxi = 0
for i in range(0,830):
    print(i)
    newset = np.append(all.copy(), i)
    #print(len(newset))
    xx = inliers[0].copy().filter(newset)
    yy = inliers_y[0]
    m = linear_model.Ridge(tol=0.0005)
    scaler = StandardScaler()
    xx = pd.DataFrame(scaler.fit_transform(xx))
    cv_results = cross_validate(m, xx, yy, cv=RepeatedKFold(n_repeats=100, n_splits=10), verbose=True, n_jobs=4)
    print('Score of ' + str(m) + 'on dataset ' + str(count))
    print("Average: " + str(np.average(cv_results['test_score'])))
    if (np.average(cv_results['test_score'])) > max:
        max = np.average(cv_results['test_score'])
        maxi = count
    print("Variance: " + str(np.var(cv_results['test_score'])))
    print(maxi)
    count = count + 1

    print(maxi)
    print(max)
'''