import pandas as pd
data = pd.read_csv("driver.csv", index_col="id")
data.head()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
kmeans.cluster_centers_
kmeans.labels_
import numpy as np
unique, counts = np.unique(kmeans.labels_, return_counts=True)
dict_data = dict(zip(unique, counts))
dict_data
import seaborn as sns
data["cluster"] = kmeans.labels_
sns.lmplot(x='mean_dist_day', y='mean_over_speed_perc', hue='cluster', palette='coolwarm', height=6, aspect=1, fit_reg=False, data=data)
kmeans.inertia_
kmeans.score
data