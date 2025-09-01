from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# loading and spliting dataset
data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Defining the weak learner
base_clf = DecisionTreeClassifier(max_depth=1)

#creating and training ada boost classifier
ada_clf = AdaBoostClassifier(base_clf, n_estimators=50, random_state=42)
ada_clf.fit(X_train, y_train)


#making prediction and evaluating accuracy 

y_pred = ada_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")