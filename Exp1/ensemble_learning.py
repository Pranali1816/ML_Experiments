from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1. Load dataset
# ---------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# ---------------------------
# 2. Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 3. Models
# ---------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

lr = LogisticRegression(max_iter=5000)
svm = SVC(probability=True)

voting = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
    voting='soft'
)

# ---------------------------
# 4. Train models
# ---------------------------
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
voting.fit(X_train, y_train)

# ---------------------------
# 5. Predictions
# ---------------------------
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)
voting_pred = voting.predict(X_test)

# ---------------------------
# 6. Accuracy scores
# ---------------------------
rf_acc = accuracy_score(y_test, rf_pred)
gb_acc = accuracy_score(y_test, gb_pred)
voting_acc = accuracy_score(y_test, voting_pred)

models = ['Random Forest', 'Gradient Boosting', 'Voting Classifier']
accuracies = [rf_acc, gb_acc, voting_acc]

# ---------------------------
# 7. Accuracy Graph
# ---------------------------
plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# ---------------------------
# 8. Confusion Matrix Plot Function
# ---------------------------
def plot_conf_matrix(cm, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ---------------------------
# 9. Confusion Matrices
# ---------------------------
cm_rf = confusion_matrix(y_test, rf_pred)
cm_gb = confusion_matrix(y_test, gb_pred)
cm_voting = confusion_matrix(y_test, voting_pred)

plot_conf_matrix(cm_rf, "Random Forest Confusion Matrix")
plot_conf_matrix(cm_gb, "Gradient Boosting Confusion Matrix")
plot_conf_matrix(cm_voting, "Voting Classifier Confusion Matrix")