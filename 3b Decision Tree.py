from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import tree
housing =fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
tree.plot_tree(dt, filled=True,feature_names=housing.feature_names)
plt.show()