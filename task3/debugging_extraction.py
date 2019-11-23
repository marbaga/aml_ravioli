import pandas as pd

X_t = pd.read_csv('extracted_features.csv', ',').iloc[:, 1:].to_numpy()
X_train = pd.read_csv('X_train.csv', ',').iloc[:, 1:].to_numpy()
print('Read')

list = pd.isnull(pd.DataFrame(X_t)).any(1).to_numpy().nonzero()[0]

print(X_train[list])
pd.DataFrame(X_train[list]).to_csv('samples_to_debug.csv')