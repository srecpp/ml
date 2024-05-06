from sklearn import datasets
cancer_data = datasets.load_breast_cancer()
print(cancer_data.data[5])
print(cancer_data.data.shape)
print(cancer_data.target)
from sklearn.model_selection import train_test_split
cancer_data = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.4,random_state=109)
from sklearn import svm
cls = svm.SVC(kernel="linear")
cls.fit(X_train,y_train)
pred = cls.predict(X_test)
from sklearn import metrics
print("acuracy:", metrics.accuracy_score(y_test,y_pred=pred))
print("recall" , metrics.recall_score(y_test,y_pred=pred))
print(metrics.classification_report(y_test, y_pred=pred))