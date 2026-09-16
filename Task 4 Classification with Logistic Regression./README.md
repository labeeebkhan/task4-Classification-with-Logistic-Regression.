# Task 4: Classification with Logistic Regression

**AI & ML Internship — Elevate Labs**
Objective: Build a binary classifier using logistic regression.

## Dataset
**Breast Cancer Wisconsin (Diagnostic) Dataset** — loaded directly via
`sklearn.datasets.load_breast_cancer()` (569 samples, 30 numeric features
describing cell nuclei from digitized breast mass images).

- **Target**: `0 = malignant` (212 cases), `1 = benign` (357 cases)
- No missing values.

## What was done
1. **Chose the dataset** — a classic, clean binary classification problem.
2. **Train/test split & standardization** — 80/20 stratified split
   (`random_state=42`), then `StandardScaler` fit on the training set only and
   applied to both sets (important since logistic regression's optimizer and
   regularization are scale-sensitive).
3. **Fit Logistic Regression** — scikit-learn's `LogisticRegression`.
4. **Evaluated** with confusion matrix, precision, recall, F1, and ROC-AUC.
5. **Tuned the decision threshold** and explained the sigmoid function.

## Results (default threshold = 0.5)

| Metric | Value |
|---|---|
| Accuracy | 0.982 |
| Precision | 0.986 |
| Recall | 0.986 |
| F1 score | 0.986 |
| ROC-AUC | 0.995 |

**Confusion matrix:**

|  | Predicted malignant | Predicted benign |
|---|---|---|
| **Actual malignant** | 41 | 1 |
| **Actual benign** | 1 | 71 |

Only 2 misclassifications out of 114 test samples — a very strong fit, which
is typical for this well-separated dataset.

### Threshold tuning
Precision, recall, and F1 were recomputed across thresholds from 0.10 to 0.95
(see `threshold_tuning.csv` / `threshold_tuning.png`). The best F1 (0.986) was
actually achieved anywhere from threshold ≈0.20–0.50, since the classes are
well separated here — recall stays at 1.0 up to threshold 0.35, then starts
dropping as the threshold rises past 0.5. In a real medical screening context
you'd likely pick a **lower** threshold than 0.5 to bias toward higher recall
(catch more true malignant cases) even at some cost to precision, since a
missed malignant case (false negative) is far costlier than a false alarm.

### Sigmoid function
See `sigmoid_function.png`. Logistic regression computes a linear score
`z = w·x + b` from the features, then squashes it into a probability with the
sigmoid function `σ(z) = 1 / (1 + e⁻ᶻ)`, which maps any real number to (0, 1).
A threshold (default 0.5) is then applied to that probability to make the
final class decision.

## Files in this repo
- `logistic_regression.py` — full script (load → split/scale → fit → evaluate → threshold tuning → plots)
- `confusion_matrix.png` — confusion matrix heatmap
- `roc_curve.png` — ROC curve (AUC = 0.995)
- `threshold_tuning.png` — precision/recall/F1 vs threshold
- `sigmoid_function.png` — sigmoid curve explanation
- `threshold_tuning.csv` — raw precision/recall/F1 values per threshold
- `metrics_summary.csv` — final metrics at threshold = 0.5

---

## Interview Questions

**1. How does logistic regression differ from linear regression?**
Linear regression predicts a **continuous** value and fits a straight line by
minimizing squared error. Logistic regression predicts the **probability of a
class** (bounded between 0 and 1) by passing a linear combination of features
through the sigmoid function, and is fit by maximizing likelihood (minimizing
log loss) rather than squared error. Linear regression's output is unbounded
and unsuitable for classification, while logistic regression's output is
always a valid probability that can be thresholded into a class label.

**2. What is the sigmoid function?**
`σ(z) = 1 / (1 + e⁻ᶻ)` — it maps any real-valued number `z` (here, the linear
combination `w·x + b`) to a value between 0 and 1, forming an S-shaped curve.
It's centered at `z = 0` (output 0.5), approaches 0 as `z → -∞`, and
approaches 1 as `z → +∞`. This lets logistic regression output something that
can be interpreted as a class probability.

**3. What is precision vs recall?**
- **Precision** = TP / (TP + FP) — "of everything I predicted positive, how
  many actually were?" High precision means few false alarms.
- **Recall** (sensitivity) = TP / (TP + FN) — "of everything that actually was
  positive, how many did I catch?" High recall means few missed cases.
There's usually a trade-off: raising the threshold increases precision but
lowers recall, and vice versa. Which one matters more is domain-specific — for
cancer detection, recall (catching every malignant case) is usually
prioritized over precision.

**4. What is the ROC-AUC curve?**
The **ROC curve** plots the True Positive Rate (recall) against the False
Positive Rate at every possible classification threshold. **AUC** (Area Under
the Curve) summarizes this into a single number between 0 and 1: it's the
probability that the model ranks a random positive example higher than a
random negative one. AUC = 0.5 means random guessing; AUC = 1.0 means perfect
separation. In this project, AUC = 0.995 — the model separates the two
classes almost perfectly. Unlike accuracy, ROC-AUC is threshold-independent
and robust to class imbalance.

**5. What is the confusion matrix?**
A table comparing predicted vs actual class labels, broken into four cells:
- **True Positive (TP)** — correctly predicted positive
- **True Negative (TN)** — correctly predicted negative
- **False Positive (FP)** — predicted positive, actually negative (Type I error)
- **False Negative (FN)** — predicted negative, actually positive (Type II error)
All the standard metrics (accuracy, precision, recall, F1) are derived from
these four counts, and it's the clearest way to see *what kind* of mistakes a
classifier is making, not just how many.

**6. What happens if classes are imbalanced?**
Accuracy becomes misleading — a model that always predicts the majority class
can score high accuracy while being useless (e.g. 95% accuracy on a 95/5
imbalanced dataset by always predicting the majority class). The model also
tends to be biased toward the majority class during training since it
minimizes overall error. Better practices: look at precision/recall/F1 and
ROC-AUC (or precision-recall AUC, which is more informative under severe
imbalance) instead of accuracy alone, use `class_weight="balanced"` in
scikit-learn, apply resampling techniques (oversampling the minority class
e.g. SMOTE, or undersampling the majority class), and choose an appropriate
decision threshold rather than defaulting to 0.5.

**7. How do you choose the threshold?**
Start from the ROC curve or precision-recall curve and pick the threshold
that best matches your cost trade-off between false positives and false
negatives, rather than defaulting to 0.5. Common approaches: maximize F1 score
(balances precision/recall equally), use Youden's J statistic (maximizes
TPR − FPR) for a balanced operating point, or set a business/domain-driven
minimum recall or precision target and pick the threshold that satisfies it
(e.g. "recall must be ≥ 0.98" for a cancer screening tool, even if that lowers
precision).

**8. Can logistic regression be used for multi-class problems?**
Yes. Two common extensions: **One-vs-Rest (OvR)** trains one binary classifier
per class (class vs. all others) and picks the class with the highest
predicted probability; **multinomial (softmax) logistic regression**
generalizes the sigmoid to the softmax function and predicts a full
probability distribution over all classes directly in one model. Scikit-learn's
`LogisticRegression` supports both via the `multi_class` parameter
(`'ovr'` or `'multinomial'`).
