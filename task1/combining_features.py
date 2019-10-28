import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import sklearn.preprocessing as preprocessing

# Code to make feature selection and prediction with Random Forest (it's actually hybrid)

X_t = pd.read_csv('task1/X_train.csv', ',').iloc[:, 1:]
y_t = pd.read_csv('task1/y_train.csv', ',').iloc[:, 1]

# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_t = filler.fit_transform(X_t, y_t)
X_t = pd.DataFrame(X_t)
print("Filled data with median")

# outlier detection
# use lof
method = LocalOutlierFactor(n_neighbors=max(50, int(0.1*X_t.shape[1])), contamination=0.2)
outlier = method.fit_predict(X_t)
target = []
for i in range(0, len(outlier)):
    if outlier[i] == -1:
        target.append(i)
X_t = X_t.copy().drop(X_t.index[target], axis=0)
y_t = y_t.copy().drop(y_t.index[target])
print("Performed outlier detection")
print(X_t.shape)

# features from images
X_train = X_t
all = pd.read_csv('task1/results/best_features.csv', ',').to_numpy().flatten()
X_train = X_train.copy().filter(all)

# features from algorithms
good = pd.read_csv('task1/results/best_features.csv', ',').to_numpy().flatten()
bad = pd.read_csv('task1/results/useless_features.csv', ',').to_numpy().flatten()

to_check = []

for i in range(0, 831):
    if i not in good and i not in bad:
        to_check.append(i)

to_check = np.array(to_check)
print('Features to check: ' + str(to_check.size))

X_t = X_t.copy().filter(to_check)


# feature selection from correlation matrix
# currently doesn't work for final prediction, because I don't know how to select features for X_test
# In basic_feature_selection this problem is solved

#insert y column
X = X_t
y = y_t
X.insert(X.shape[1], 'y', y, True)

cor = pd.DataFrame(X).corr()
columns = np.full((cor.shape[0],), True, dtype=bool)
for i in range(cor.shape[0]):
    for j in range(i+1, cor.shape[0]):
        if abs(cor.iloc[i,j]) > 0.85 or abs(cor.iloc[j,cor.shape[0]-1]) < 0.03:
            if columns[j]:
                columns[j] = False
data = X.iloc[:, columns]
#removing y
X_t = data.iloc[:, :data.shape[1]-1]
print('Shape of the cleaned matrix after correlation selection: ' + str(X_t.shape))
# end of feature selection from correlation matrix


X_t = pd.DataFrame(X_t)


# Random forest feature selection
#model = rfr_model(X_t, y_t)    # possible other model
#sel = SelectFromModel(model)
sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
sel.fit(X_t, y_t)
selected_feat = X_t.columns[(sel.get_support())]
X_t = X_t.filter(selected_feat)
print('Number of selected features: ' + str(len(selected_feat)))
while len(selected_feat) > 130:
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(X_t, y_t)
    selected_feat = X_t.columns[(sel.get_support())]
    X_t = X_t.filter(selected_feat)
    print('Number of selected features: ' + str(len(selected_feat)))
# end


#print(good)
#print(selected_feat)

# merging features from images and from algorithms
print('Size of X_train: ' + str(X_train.shape))
print('Size of X_t: ' + str(X_t.shape))
X_t = pd.concat([X_train.reset_index(drop=True), X_t.reset_index(drop=True)], axis=1)
print('Size of the merged table: ' + str(X_t.shape))

# normalizing
poly = preprocessing.PolynomialFeatures(2)
scaler = preprocessing.StandardScaler()
X_t = pd.DataFrame(scaler.fit_transform(X_t))

#ElasticNetCV performs automatic hyperparameter search. Lasso models works a little better.
#we should try different linear models
#model = linear_model.ElasticNetCV(l1_ratio=0.5, eps=1e-3, n_alphas=10, cv=10, selection='random')
#model = linear_model.LassoCV(cv=5)
#model = linear_model.RidgeCV(cv=10)
model = RandomForestRegressor(max_depth=6, random_state=False, n_estimators=50)
#model = rfr_model(X_t, y_t)

model.fit(X_t, y_t.ravel())
print(model.score(X_t, y_t))

cv_results = cross_validate(model, X_t, y_t, cv=5, scoring='r2')
print('Score of ' + str(model) + ': ')
print(cv_results['test_score'])
print("Average: " + str(np.average(cv_results['test_score'])))

#pd.DataFrame(X_t).to_csv('task1/results/features_of_combining_features.csv', ',')
#pd.DataFrame(y_t).to_csv('task1/results/features_of_combining_features_y.csv', ',')

# PREDICTION
print('Preparing for prediction')

X_test = pd.read_csv('task1/X_test.csv', ',').iloc[:, 1:]
print(X_test)
# fill with median
filler = SimpleImputer(missing_values=np.nan, strategy='median')
X_test = filler.fit_transform(X_test)
X_test = pd.DataFrame(X_test)
#print(X_test)
print("Filled test with median")
#print(X_test)
# good is the array of features selected manually, selected_feat of the ones selected with random forest
useful_features = np.concatenate((good, selected_feat))
print("Size before eliminating duplicates: " + str(useful_features.size))
useful_features = np.unique(useful_features)
useful_features.sort()  # quite useless actually, but I liked it
print("Size after eliminating duplicates: " + str(useful_features.size))
X_test = X_test.copy().filter(useful_features)
print("Performed feature selection. " + str(X_test.shape) + ' is the final shape for the test matrix.')
scaler = StandardScaler()
X_test = pd.DataFrame(scaler.fit_transform(X_test))
#print(X_test)
print("Standardized test samples")

print("Predicting...")

predictions = model.predict(X_test)

print("Computing predictions")

answer = pd.read_csv('task1/X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=predictions, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('task1/results/random forest_2.csv', ',', index=False)

print("Prediction formattes and written to file")
