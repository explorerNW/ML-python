from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

x = [
    [1, 2, 3],
    [11, 12, 13]
]

y = [0 ,1]

clf.fit(x, y)
result = clf.predict(x)
print(result)

pred = clf.predict(
    [
        [123, 2441, 35463],
        [7, 234, 5675]
    ]
)
print('predict: ', pred)

from sklearn.preprocessing import StandardScaler

x = [
    [0, 15],
    [1, -10]
]
result = StandardScaler().fit(x).transform(x)
print(result)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


pipe = make_pipeline(StandardScaler(), LogisticRegression())
import os
import pandas as pd
file_path = os.path.join(os.path.dirname(__file__), 'iris.data')
df = pd.read_csv(file_path, header=None, encoding='utf-8')
df.tail()

# X, y = load_iris(return_X_y=True)
X = df.iloc[0:200, [0, 2]].values
y = df.iloc[0:200, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipe.fit(X_train, y_train)
result = accuracy_score(pipe.predict(X_test), y_test)
print(result)

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

# X, y = make_regression(n_samples=1000, n_features=1000, random_state=0)
y = df.iloc[0:200, 3].values
lr = LinearRegression()
result = cross_validate(lr, X, y)
print('result = ', result)

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
param_distributions = { 'n_estimators': randint(1, 5), 'max_depth': randint(5, 10) }
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5, param_distributions=param_distributions, random_state=0)
search.fit(X_train, y_train)
print('search: ', search.best_params_)
print('score: ', search.score(X_test, y_test))

from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

from sklearn import linear_model
lreg = linear_model.LinearRegression()
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lreg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
y_pred = lreg.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

import torch
x = torch.rand(5, 3)
print(x)



