import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn import feature_selection
from sklearn import metrics
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, ExtraTreesClassifier
from scipy.stats import shapiro
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import random

import warnings

def select_features_by_corr(max_corr, X, y):

    X2 = pd.DataFrame(X.copy())

    # insert y column
    X2.insert(X2.shape[1], 'y', y.copy(), True)

    # feature selection
    cor = pd.DataFrame(X2).corr()
    columns = np.full((cor.shape[0],), True, dtype=bool)
    for i in range(cor.shape[0]):
        if columns[i]:
            for j in range(i + 1, cor.shape[0]):
                if abs(cor.iloc[i, j]) > max_corr:
                    columns[j] = False

    to_keep = []
    for i in range(cor.shape[0]):
        if columns[i]:
            to_keep.append(i)

    return to_keep


m = ensemble.GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=2500, subsample=0.8, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',random_state=10)

X_t = pd.read_csv('task1/X_train.csv', ',').iloc[:, 1:].to_numpy()
y_t = pd.read_csv('task1/y_train.csv', ',').iloc[:, 1].to_numpy()
ind = pd.read_csv('not_uniform_features.csv').iloc[:, 1].to_numpy()

X_train, X_val, y_train, y_val = model_selection.train_test_split(X_t, y_t, test_size=0.2, random_state=1)

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_train = filler.fit_transform(X_train, y_train)
X_train = pd.DataFrame(X_train)
print("Filled data with median")

# use lof for outlier detection
method = LocalOutlierFactor(n_neighbors=max(50, int(0.1*X_t.shape[1])), contamination=0.05, n_jobs=-1)
outlier = method.fit_predict(X_train)
target = []
for i in range(0, len(outlier)):
    if outlier[i] == -1:
        target.append(i)
X_train = X_train.copy().drop(X_train.index[target], axis=0)
y_train = y_train.copy().drop(y_train.index[target])
print("Performed outlier detection")
print(X_train.shape)

X_train = X_train.filter(ind)
print(X_train.shape)
sel = VarianceThreshold()
X_train = pd.DataFrame(sel.fit_transform(X_train))
print(X_train.shape)
print("Eliminated irrelevant features")

#to_keep_corr = select_features_by_corr(0.95, X_train, y_train)
#X_train = pd.DataFrame(X_train).filter(to_keep_corr)
#print(X_train.shape)
#print("Eliminated highly correlated features")

#selector = feature_selection.RFE(xgb.XGBRegressor(), step=100, n_features_to_select = 200, verbose=True)
#selector = selector.fit(X_train, y_train.to_numpy().ravel())
#to_keep = selector.get_support(indices=True)

forest = RandomForestRegressor(n_estimators=200) # best results at 200
forest.fit(X_train, y_train.to_numpy().ravel())
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
to_keep = indices[:220]

X_train = X_train.filter(to_keep)
print(X_train.shape)


for i in range(0, X_train.shape[1]):
    print(shapiro(X_train.iloc[:, i]))
    #plt.scatter(X_train.iloc[:, i], y_train)
    plt.hist(X_train.iloc[:, i], bins=100)
    plt.show()

print("Performed feature selection. " + str(X_train.shape) + ' is the final shape for the training matrix.')

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
print("Standardized samples")

warnings.simplefilter(action='ignore', category=DeprecationWarning)
print('Preparing for prediction')

# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_val = filler.fit_transform(X_val)
X_val = pd.DataFrame(X_val)
print("Filled test with median")
X_val = X_val.filter(ind)
sel = VarianceThreshold()
X_val = pd.DataFrame(sel.fit_transform(X_val))
#X_val = X_val.filter(to_keep_corr)
X_val = X_val.filter(to_keep)
print("Performed feature selection. " + str(X_val.shape) + ' is the final shape for the validation matrix.')
X_val = pd.DataFrame(scaler.transform(X_val))
print("Standardized test samples")

#model = xgb.XGBRegressor()
#model.fit(X_train, y_train)
#pred = model.predict(X_val)
#print(metrics.r2_score(y_val, pred))

m.fit(X_train, y_train.to_numpy().ravel())
print("CV score: ")
print(np.average(model_selection.cross_val_score(m,X_train, y_train.to_numpy().ravel(), cv=5, n_jobs=-1)))
predictions = m.predict(X_val)
print("Validation score: ")
plt.scatter(predictions, y_val)
plt.show()
print(metrics.r2_score(y_val, predictions))

'''
ind = []
for i in range(0, X_t.shape[1]):
    print(i)
    val = X_t[:, i]
    n, bins, a = plt.hist(X_t[:, i])
    if(2*n[9]<n[5]):
        plt.clf()
        plt.scatter(X_t[:, i], y_t)
        #plt.show()
        ind.append(i)
    plt.clf()
pd.DataFrame(ind).to_csv('not_uniform_features.csv')


added_features = []
found = True
while len(to_keep)<200 or found:
    found = False
    xx = X_train.copy()
    yy = y_train.copy()

    xx = xx.filter(to_keep)
    scaler = StandardScaler()
    xx = pd.DataFrame(scaler.fit_transform(xx))
    m.fit(xx, yy.to_numpy().ravel())

    xv = X_val.copy()
    filler = SimpleImputer(missing_values=np.nan, strategy='median')
    xv = filler.fit_transform(xv)
    xv = pd.DataFrame(xv)
    xv = xv.filter(ind)
    sel = VarianceThreshold()
    xv = pd.DataFrame(sel.fit_transform(xv))
    xv = xv.filter(to_keep)
    print("Baseline shape is " + str(xv.shape))
    xv = pd.DataFrame(scaler.transform(xv))

    pred = m.predict(xv)
    baseline = metrics.r2_score(y_val, pred)
    print("Baseline score is " + str(baseline))

    lis = list(range(X_train.shape[1]))
    random.shuffle(lis)
    for i in lis:
        if not i in to_keep:
            print("Checking feature " + str(i) + ". Added features: " + str(added_features))
            newset = np.append(to_keep.copy(), i)
            xx = X_train.copy().filter(newset)
            scaler = StandardScaler()
            xx = pd.DataFrame(scaler.fit_transform(xx))
            m.fit(xx, yy.to_numpy().ravel())

            xv = X_val.copy()
            filler = SimpleImputer(missing_values=np.nan, strategy='median')
            xv = filler.fit_transform(xv)
            xv = pd.DataFrame(xv)
            xv = xv.filter(ind)
            sel = VarianceThreshold()
            xv = pd.DataFrame(sel.fit_transform(xv))
            xv = xv.filter(newset)
            xv = pd.DataFrame(scaler.transform(xv))

            pred = m.predict(xv)
            new_result = metrics.r2_score(y_val, pred)
            print("Latest result: " + str(new_result) + ", number of features is " + str(len(newset)))
            if new_result > baseline*1.002:
                to_keep = np.append(to_keep, i)
                added_features = np.append(added_features, i)
                found = True
                break
'''