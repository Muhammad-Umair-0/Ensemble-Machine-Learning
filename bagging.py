from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# loading and spliting the dataset

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating a base classifier

base_clf = DecisionTreeClassifier()

# creating and fitting the bagging classifier

bagging_clf  = BaggingClassifier(base_clf, n_estimators=10, random_state=42)
bagging_clf.fit(X_train, y_train)


#making prediction and Evaluating the model

y_pred = bagging_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))