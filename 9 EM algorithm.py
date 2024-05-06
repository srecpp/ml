from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Plot real data
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X[:, 2], X[:, 3], c=colormap[y])

# Apply Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm = gmm.predict(X)

# Plot GMM classification
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X[:, 2], X[:, 3], c=colormap[y_cluster_gmm])

# Evaluation
print('The accuracy score of EM: ', metrics.accuracy_score(y, y_cluster_gmm))
print('The Confusion matrix of EM:\n', metrics.confusion_matrix(y, y_cluster_gmm))

plt.show()
