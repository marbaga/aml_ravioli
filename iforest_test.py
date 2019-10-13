#THIS SCRIPT WAS USED TO EXPERIMENT ISOLATION FOREST OUTLIER DETECTION. IT PROVED TO BE UNDERPERFORMING

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model
from sklearn.model_selection import cross_validate, KFold, RepeatedKFold, LeaveOneOut, ShuffleSplit
from sklearn.preprocessing import StandardScaler


def select_features_by_corr(max_corr, min_corr, X, y):
    X = pd.DataFrame(X)
    X2 = X.copy()

    # insert y column
    X2.insert(X2.shape[1], 'y', y.copy(), True)

    # eliminate features with variance 0
    sel = VarianceThreshold(threshold=0)
    X2 = sel.fit_transform(X2)

    # feature selection
    cor = pd.DataFrame(X2).corr()
    columns = np.full((cor.shape[0],), True, dtype=bool)
    for i in range(cor.shape[0]):
        for j in range(i + 1, cor.shape[0]):
            if abs(cor.iloc[i, j]) > max_corr or abs(cor.iloc[j, cor.shape[0] - 1]) < min_corr:
                if columns[j]:
                    columns[j] = False
    selected_columns = pd.DataFrame(X2).columns[columns]
    data = pd.DataFrame(X2[:, selected_columns])

    # removing y
    data = data.iloc[:, :data.shape[1] - 1]
    print(data)

    return data


def isolation_forest(X, y):
    # compute rank according to isolation forests
    iterations = 50
    result = np.zeros(X.shape[0])

    count = 0
    for i in range(0, iterations):
        ifor = IsolationForest(contamination=0.1, behaviour='new')
        ifor.fit(X, y)
        print(np.average(ifor.score_samples(X)))
        outlier = ifor.score_samples(X)
        #plt.show()
        #print(str(sum(i < -0.49 for i in ifor.score_samples(X))))
        for j in range(0, outlier.shape[0]):
            if outlier[j] <= -0.48: #0.49
                count = count + 1
                #print('outlier: ' + str(j) + ' ' + str(ifor.score_samples(X)[j]))
                result[j] = result[j] + 1
    n = count / iterations
    return result, n


X_t = pd.read_csv('task1/X_train.csv', ',').iloc[:, 1:]
y_t = pd.read_csv('task1/y_train.csv', ',').iloc[:, 1]

# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_t = filler.fit_transform(X_t, y_t)
X_t = pd.DataFrame(X_t)

print("Filled data with median")

inliers = []
inliers_y = []

#computing isolation forest based ranking for outliers
result, n = isolation_forest(X_t, y_t)
#plt.hist(result)
#plt.show()

#trying out different numbers of outliers
for mult in [2, 1, 0.5]:
    # compute X removing mult*n outliers
    ind = np.argpartition(result, int(-1 * mult * n))[int(-1 * mult * n):]
    ind = np.sort(ind)

    inliers.append(X_t.copy().drop(X_t.index[ind], axis = 0))
    inliers_y.append(y_t.copy().drop(y_t.index[ind]))
print("Added datasets calculated with IF")

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

#these commented lines were used to compute subset of the set of features. amongst those, only the ones with best resutls were chosen
'''
for l in range (0, len(inliers)):
    for s in range (2, 100):
        #selected.append(inliers)
        scaler = StandardScaler()
        estimator = linear_model.Ridge()
        selector = RFECV(estimator, step=s, cv=10, verbose=True)
        selector = selector.fit(scaler.fit_transform(inliers[l].copy()), inliers_y[l])
        res = inliers[l]
        to_keep = selector.get_support(indices=True)
        pd.DataFrame(to_keep).to_csv('task1/results/to_keep' + str(s) + '.csv', ',', index=False)

        #print(to_keep)
        res = res.filter(to_keep)
        #print(res)
        selected.append(res)
        selectedy.append(inliers_y[l])
'''

all = []
#these are the subsets that performed the best. they are joined together
for i in [8, 13, 52, 59, 75, 83, 93, 95]:
    print(i)
    new = pd.read_csv('task1/results/to_keep'+str(i)+'.csv', ',')
    all.extend(new.to_numpy().flatten())
    print(new.shape)

set2 = set(all)
set2.add(0)
all = np.sort(list(set2))
print(len(all))

#these features were selected empirically
all = np.append(all, 786)
all = np.append(all, 386)
all = np.append(all, 423)
print(len(all))
selected.append(inliers[0].copy().filter(all))
selectedy.append(inliers_y[0])



#these last lines help score the results of the different datasets produced by different outlier and feature selection policies.
count = 0
max = 0
maxi = 0
arr = [linear_model.Ridge(tol=0.0005), linear_model.RidgeCV(alphas=[0.01, 0.05, 0.1, 0.5, 1, 5], cv=10)]
for k in range(0, len(selected)):
    for m in arr:
        scaler = StandardScaler()
        selected[k] = pd.DataFrame(scaler.fit_transform(selected[k]))
        cv_results = cross_validate(m, selected[k], selectedy[k], cv=RepeatedKFold(n_repeats=100, n_splits=10), verbose=True, n_jobs=4)
        print('Score of ' + str(m) + 'on dataset ' + str(count))
        #print(cv_results['test_score'])
        print("Average: " + str(np.average(cv_results['test_score'])))
        if(np.average(cv_results['test_score'])) > max:
            max = np.average(cv_results['test_score'])
            maxi = count
        print("Variance: " + str(np.var(cv_results['test_score'])))
        print(maxi)
        count = count + 1

print(maxi)
print(max)

#try deactivating scaler when using lofCV