from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all', categories=None, shuffle=True, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data.data)

k = 20
kmeans = KMeans(n_clusters=k, random_state=42)  # Fixed variable name here
kmeans.fit(tfidf)

print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()

new_doc = ["car engine performance", "computer science programming"]
new_tfidf = tfidf_vectorizer.transform(new_doc)
predicted_cluster = kmeans.predict(new_tfidf)

print("\nPredicted clusters for new documents:")
for i, doc in enumerate(new_doc):
    print("Document:", doc, "- Predicted Cluster:", predicted_cluster[i])
