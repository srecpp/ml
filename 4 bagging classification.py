import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
bc = datasets.load_breast_cancer()
X = bc.data
y= bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1,
stratify=y)
pipeline = make_pipeline(StandardScaler(),LogisticRegression(random_state=1))
bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
max_features=10,max_samples=100, random_state=1, n_jobs=5)
bgclassifier.fit(X_train, y_train)
print('Model test Score: %.3f, ' %bgclassifier.score(X_test, y_test), 'Model training Score:%.3f' %bgclassifier.score(X_train, y_train))