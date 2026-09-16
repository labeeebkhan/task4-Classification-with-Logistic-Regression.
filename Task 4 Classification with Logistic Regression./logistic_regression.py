"""
Task 4: Classification with Logistic Regression
AI & ML Internship - Elevate Labs

Objective: Build a binary classifier using logistic regression.
Tools: Scikit-learn, Pandas, Matplotlib

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset - built into scikit-learn,
so it's fully reproducible with no external download. Target: malignant (0) vs
benign (1) tumor diagnosis based on cell nuclei measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    classification_report,
)

plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------
# 1. Choose a binary classification dataset
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load dataset")
print("=" * 60)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# target: 0 = malignant, 1 = benign
print(f"Shape: {df.shape}")
print(f"Target classes: {dict(zip(data.target_names, range(2)))}")
print(f"Class balance:\n{df['target'].value_counts()}")
print(f"\nMissing values: {df.isnull().sum().sum()}")

X = df.drop(columns=["target"])
y = df["target"]

# ---------------------------------------------------------------
# 2. Train/test split and standardize features
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Train-test split & standardization")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized (mean=0, std=1) using StandardScaler fit on train set only.")

# ---------------------------------------------------------------
# 3. Fit a Logistic Regression model
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Fit Logistic Regression")
print("=" * 60)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of class "1" (benign)

# ---------------------------------------------------------------
# 4. Evaluate: confusion matrix, precision, recall, ROC-AUC
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Evaluation (default threshold = 0.5)")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")
print(f"\nConfusion matrix:\n{cm}")
print(f"\nFull classification report:\n{classification_report(y_test, y_pred, target_names=data.target_names)}")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix (threshold = 0.5)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved plot -> confusion_matrix.png")

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
print("Saved plot -> roc_curve.png")

# ---------------------------------------------------------------
# 5. Tune threshold & explain sigmoid function
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Threshold tuning")
print("=" * 60)

thresholds_to_try = np.arange(0.1, 1.0, 0.05)
results = []
for t in thresholds_to_try:
    y_pred_t = (y_proba >= t).astype(int)
    results.append(
        {
            "threshold": round(t, 2),
            "precision": precision_score(y_test, y_pred_t, zero_division=0),
            "recall": recall_score(y_test, y_pred_t, zero_division=0),
            "f1": f1_score(y_test, y_pred_t, zero_division=0),
        }
    )
thresh_df = pd.DataFrame(results)
thresh_df.to_csv("threshold_tuning.csv", index=False)
print(thresh_df.to_string(index=False))

best_row = thresh_df.loc[thresh_df["f1"].idxmax()]
print(f"\nBest threshold by F1 score: {best_row['threshold']} (F1={best_row['f1']:.4f})")

# Plot precision/recall/F1 vs threshold
plt.figure(figsize=(8, 6))
plt.plot(thresh_df["threshold"], thresh_df["precision"], marker="o", label="Precision")
plt.plot(thresh_df["threshold"], thresh_df["recall"], marker="o", label="Recall")
plt.plot(thresh_df["threshold"], thresh_df["f1"], marker="o", label="F1 score")
plt.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Default threshold (0.5)")
plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Classification Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_tuning.png")
plt.close()
print("Saved plot -> threshold_tuning.png")

# Sigmoid function plot (explains how logistic regression maps scores -> probabilities)
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))
plt.figure(figsize=(7, 5))
plt.plot(z, sigmoid, color="purple", linewidth=2)
plt.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
plt.axvline(0, color="gray", linestyle="--", alpha=0.7)
plt.xlabel("z (linear combination of features: w·x + b)")
plt.ylabel("sigmoid(z) = predicted probability")
plt.title("The Sigmoid Function")
plt.tight_layout()
plt.savefig("sigmoid_function.png")
plt.close()
print("Saved plot -> sigmoid_function.png")

# ---------------------------------------------------------------
# 6. Save metrics summary
# ---------------------------------------------------------------
summary = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Value": [acc, prec, rec, f1, auc],
    }
)
summary.to_csv("metrics_summary.csv", index=False)
print("\nSaved metrics -> metrics_summary.csv")
print("\nDone.")
