
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

X_t = pd.read_csv('task1/X_train.csv', ',')
y_t = pd.read_csv('task1/y_train.csv', ',')

pipe = Pipeline([
                 ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean'))
                 ])

pipe = pipe.fit(X_t, y_t)

X_t_filled = pipe.named_steps['imputer'].transform(X_t)

pd.DataFrame(X_t_filled[0:, 1:]).to_csv('task1/output.csv', ',')
