import pandas as pd
import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
import warnings
from sklearn.feature_selection import RFE
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import GradientBoostingRegressor

#Performs manual feature selection and prediction

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
method = LocalOutlierFactor(n_neighbors=max(50, int(0.1*X_t.shape[1])), contamination=0.1)
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

# normalizing
scaler = preprocessing.StandardScaler()
X_t = pd.DataFrame(scaler.fit_transform(X_t))

X_test = pd.DataFrame(X_test)
X_test = pd.DataFrame(scaler.transform(X_test))

ans = []
first = True
n=0
for i in range(0,42):
    n = n+1
    model = GradientBoostingRegressor(learning_rate=0.01, max_depth=4, n_estimators=3500, subsample=0.8, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',random_state=i)
    model.fit(X_t, y_t.ravel())

    predictions = model.predict(X_test)
    if(first):
        ans = predictions
    else:
        for j in range(0, len(predictions)):
            ans[j] = ans[j] + predictions[j]
    print(predictions)
    print(i)
    first = False

for i in range(0, len(ans)):
    ans[i] = ans[i]/n
    if np.abs(ans[i] - int(round(ans[i]))) < 0.2:
        ans[i] = int(round(ans[i]))

answer = pd.read_csv('task1/X_test.csv', ',')[['id']]
answer = pd.concat([answer, pd.DataFrame(data=ans, columns=['y'])], axis=1)
pd.DataFrame(answer).to_csv('task1/results/RFE_200.csv', ',', index=False)

print("Prediction formattes and written to file")