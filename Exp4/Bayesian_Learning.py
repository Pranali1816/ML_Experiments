from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------
# Load dataset
# --------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Model (Bayesian Learning)
# --------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# --------------------------
# Prediction
# --------------------------
y_pred = model.predict(X_test)

# --------------------------
# Accuracy
# --------------------------
acc = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", acc)

# --------------------------
# Confusion Matrix
# --------------------------
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm, cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()