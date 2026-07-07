import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes 2.csv")
FEATURES = ["BMI", "Age", "Glucose", "BloodPressure"]
X = df[FEATURES]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_s, y_train)
log_pred = log_model.predict(X_test_s)
log_proba = log_model.predict_proba(X_test_s)[:, 1]

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_s, y_train)
tree_pred = tree_model.predict(X_test_s)
tree_proba = tree_model.predict_proba(X_test_s)[:, 1]

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_pred))
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
log_scores = [
    accuracy_score(y_test, log_pred),
    precision_score(y_test, log_pred),
    recall_score(y_test, log_pred),
    f1_score(y_test, log_pred),
    roc_auc_score(y_test, log_proba)
]
tree_scores = [
    accuracy_score(y_test, tree_pred),
    precision_score(y_test, tree_pred),
    recall_score(y_test, tree_pred),
    f1_score(y_test, tree_pred),
    roc_auc_score(y_test, tree_proba)
]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar([i - width/2 for i in x], log_scores, width, label="Logistic Regression", color="#DC2626")
plt.bar([i + width/2 for i in x], tree_scores, width, label="Decision Tree", color="#1E3A8A")
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Logistic Regression vs Decision Tree — Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_chart.png", dpi=200)
print("Saved comparison_chart.png")
fpr_log, tpr_log, _ = roc_curve(y_test, log_proba)
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr_log, tpr_log, color="#DC2626", linewidth=2,
         label=f"Logistic Regression (AUC = {roc_auc_score(y_test, log_proba):.3f})")
plt.plot(fpr_tree, tpr_tree, color="#1E3A8A", linewidth=2,
         label=f"Decision Tree (AUC = {roc_auc_score(y_test, tree_proba):.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression vs Decision Tree")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)
print("Saved roc_curve.png")