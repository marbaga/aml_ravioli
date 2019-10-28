import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.linalg import LinAlgWarning
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
from sklearn.model_selection import cross_validate, KFold, RepeatedKFold, LeaveOneOut, ShuffleSplit
from sklearn.preprocessing import StandardScaler
import warnings
from xgboost import XGBRegressor
from estimator import InputTransform
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
import sklearn.preprocessing as preprocessing
from rfr_model import rfr_model
from sklearn.feature_selection import VarianceThreshold
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from xbregressor_tuning import xbr_model
from matplotlib import pyplot as plt
# It's a collage from previous code, with some new things. The X_test is
# modified together with X_train, otherwise I didn't know ho to distinguish features
# in the correlation matrix, that's why you will sometimes find it with apparently no meaning

#suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=LinAlgWarning)

X_t = pd.read_csv('task1/X_train.csv', ',').iloc[:, 1:]
y_t = pd.read_csv('task1/y_train.csv', ',').iloc[:, 1]
X_test = pd.read_csv('task1/X_test.csv', ',').iloc[:, 1:]


# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_t = filler.fit_transform(X_t, y_t)
X_t = pd.DataFrame(X_t)
print("Filled data with median")
#X_test = np.asarray(X_test)
X_test = filler.transform(X_test)
X_test = pd.DataFrame(X_test)

# outlier detection
# use lof
method = LocalOutlierFactor(n_neighbors=max(50, int(0.1*X_t.shape[1])), contamination=0.15)
outlier = method.fit_predict(X_t)
target = []
for i in range(0, len(outlier)):
    if outlier[i] == -1:
        target.append(i)
X_t = X_t.copy().drop(X_t.index[target], axis=0)
y_t = y_t.copy().drop(y_t.index[target])
print("Performed outlier detection")


# features from images obtained manually, will be merged with the others at the end
X_train = X_t
best = pd.read_csv('task1/results/best_features_2.csv', ',').to_numpy().flatten()
X_train = X_train.copy().filter(best)
X_test_first = X_test.copy().filter(best)

print(X_train.shape)
print(X_test_first.shape)

# manually obtained features (good and bad), to remove so that now the feature selection is concentrated
# in the medium ones
good = pd.read_csv('task1/results/best_features_2.csv', ',').to_numpy().flatten()
bad = pd.read_csv('task1/results/useless_features.csv', ',').to_numpy().flatten()

to_check = []   #list of feature to consider

for i in range(0, 831):
    if i not in bad:
        to_check.append(i)

to_check = np.array(to_check)
print('Features to check: ' + str(to_check.size))

X_t = X_t.copy().filter(to_check)
X_test = X_test.copy().filter(to_check)

# feature selection from correlation matrix
#insert y column
X = X_t
y = y_t
X.insert(X.shape[1], 'y', y, True)
test_data = X_test.insert(X_test.shape[1], 'y', y, True)

cor = pd.DataFrame(X).corr()
columns = np.full((cor.shape[0],), True, dtype=bool)
for i in range(cor.shape[0]):
    for j in range(i+1, cor.shape[0]):
        if abs(cor.iloc[i,j]) > 0.9 or abs(cor.iloc[j,cor.shape[0]-1]) < 0.03:
            if columns[j]:
                columns[j] = False
data = X.iloc[:, columns]
test_data = X_test.iloc[:, columns]
#removing y
X_t = data.iloc[:, :data.shape[1]-1]
X_test = test_data.iloc[:, :data.shape[1]-1]
print('Shape of the cleaned matrix after correlation selection: ' + str(X_t.shape))
print('Shape of the test cleaned matrix after correlation selection: ' + str(X_t.shape))
# end of feature selection from correlation matrix

'''# RFECV
#model = Ridge()
model = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=2500, subsample=0.8, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',random_state=10)
selector = RFECV(estimator=model, step=10, min_features_to_select=40, cv=10)
selector = selector.fit(X_t, y_t)
support = selector.get_support()


# Trying to select a good number of features to approximate 200 total features
v = []
for i in range(0,8):
    v.append(0)
i=0
while i < selector.ranking_.size:
    for j in range(0,8):
        if selector.ranking_[i] == j:
            v[j] = v[j]+1
    i = i+1
print(v)
num_features = 0
n_to_stop = 0
i = 0
while num_features < 95:
    num_features = num_features + v[i]
    n_to_stop = n_to_stop+1
    i = i+1

# collects the features with good enough ranking
useful_features = []
i=0
while i < selector.ranking_.size:
    if selector.ranking_[i] < n_to_stop:
        useful_features.append(i)
    i = i+1

useful_features = np.array(useful_features)

X_t = X_t.iloc[:, support]
X_test = X_test.iloc[:, support]

print(X_t.shape)
print(X_test.shape)
#end feature selection with rfecv
'''

# RFE feature selection with no cross validation
model = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=2500, subsample=0.8, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',random_state=10)
selector = RFE(estimator=model, n_features_to_select=61, step=10)
selector = selector.fit(X_t, y_t)
support = selector.get_support()
X_t = X_t.iloc[:, support]
X_test = X_test.iloc[:, support]
'''
for i in range(0,X_t.shape[1]):
    plt.scatter(X_t.iloc[:,i], y_t)
    plt.show()'''

print(X_t.shape)
print(X_test.shape)
'''
# merging features from images and from algorithms, for both train and test
print('Size of X_train: ' + str(X_train.shape))
print('Size of X_t: ' + str(X_t.shape))
X_t = pd.concat([X_train.reset_index(drop=True), X_t.reset_index(drop=True)], axis=1)
print('Size of the merged table: ' + str(X_t.shape))

print('Size of X_test_first: ' + str(X_test_first.shape))
print('Size of X_test: ' + str(X_test.shape))
X_test = pd.concat([X_test_first.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)
print('Size of the merged table: ' + str(X_test.shape))

pd.DataFrame(X_t).to_csv('task1/results/merged_table.csv', ',', index=True)
pd.DataFrame(y_t).to_csv('task1/results/merged_table_y.csv', ',', index=True)
'''


# normalizing
scaler = preprocessing.StandardScaler()
X_t = pd.DataFrame(scaler.fit_transform(X_t))


# POSSIBLE MODELS
#ElasticNetCV performs automatic hyperparameter search. Lasso models works a little better.
#we should try different linear models
#model = linear_model.ElasticNetCV(l1_ratio=0.5, eps=1e-3, n_alphas=10, cv=10, selection='random')
#model = linear_model.LassoCV(cv=5)
#model = linear_model.RidgeCV(cv=10)
#model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
#model = rfr_model(X_t, y_t)
#model = GradientBoostingRegressor(learning_rate=0.01, max_depth=5, n_estimators=2500, subsample=0.8, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',random_state=10)
#model = xbr_model(X_t, y_t)
#model = XGBRegressor(learning_rate=0.1, max_depth=3, subsample=0.7, n_estimators=100, colsample_bytree=0.4, alpha=0.1)
model.fit(X_t, y_t.ravel())
print(model.score(X_t, y_t))

cv_results = cross_validate(model, X_t, y_t, cv=5, scoring='r2')
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))

# PREDICTION
print('Preparing for prediction')

#print(X_test)
# fill with median
#filler = SimpleImputer(missing_values=np.nan, strategy='median')    #are there outliers? I have used mean for the submission
#X_test = filler.transform(X_t)
X_test = pd.DataFrame(X_test)
#print(X_test)
print("Filled test with median")
#print(X_test)
print("Performed feature selection. " + str(X_test.shape) + ' is the final shape for the test matrix.')
X_test = pd.DataFrame(scaler.transform(X_test))
#print(X_test)
print("Standardized test samples")

print("Predicting...")

predictions = model.predict(X_test)

print("Computing predictions")

answer = pd.read_csv('task1/X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=predictions, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('task1/results/RFE_200.csv', ',', index=False)

print("Prediction formattes and written to file")
