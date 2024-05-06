import pandas as pd
df = pd.read_csv("movies.csv")
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
documents = df['overview'].values.astype("U")
documents
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)
k = 30
model = KMeans(n_clusters=k, init = 'k-means++',max_iter = 100, n_init=1)
model.fit(features)
df['cluster']=model.labels_
df.head()
clusters = df.groupby('cluster')
for cluster in clusters.groups:
    f=open('cluster'+str(cluster)+'.csv','w')
    data = clusters.get_group(cluster)[['title','overview']]
    f.write(data.to_csv(index_label='id'))
f.close()
print("Cluster centroids: \n")
order_centroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names_out()
for i in range(k):
    print("Cluster %d:" %i)
    for j in order_centroids[i, :10]: 
        print(' %s' %terms[j])
    print(' ')