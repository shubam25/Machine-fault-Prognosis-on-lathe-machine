import pandas as pd
import numpy as np
from sklearn import svm
new_final = pd.read_csv('final_data.csv')
new_final.head() 
print(new_final);
new_final.drop('Unnamed: 0', axis=1, inplace=True)
x = new_final.drop('y', axis=1)
y = new_final['y']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
print(X_train.shape);
#applying logistic regression
from sklearn.linear_model import LogisticRegression
clf_lr=LogisticRegression()
print(clf_lr.fit(X_train, y_train));
clf_lr.score(X_train, y_train)
print(clf_lr.score(X_test, y_test));
#applying support vector machine
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# applying grid search for hyper-parameter tuning
svc_param_grid = {'C': [1, 10, 100, 1000],
                  'gamma': [0.01, 0.001, 0.0001],
                  'kernel': ['rbf','linear','poly']
                 }
model_svc2 = SVC()
grid_model_svc2 = GridSearchCV(model_svc2, svc_param_grid, cv=None)
print(grid_model_svc2.fit(X_train, y_train))
print(grid_model_svc2.score(X_train, y_train))
print(grid_model_svc2.score(X_test, y_test))
grid_model_svc2.best_estimator_
y_pred = grid_model_svc2.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))