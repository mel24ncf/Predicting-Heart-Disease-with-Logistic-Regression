# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, recall_score, precision_score, accuracy_score, 
                             classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay, 
                             f1_score, precision_recall_fscore_support)
from util import plot_confusion_matrix, plot_roc_curve, plot_threshold_analysis
import pickle

# Read in test data
X_test, y_test = pd.read_pickle("Data/test_data.pkl")
X_val, y_val = pd.read_pickle("Data/val_data.pkl")

# Load trained logistic regression model
with open("logistic_regression_model.pkl", "rb") as pfile:
    pipeline = pickle.load(pfile)

# Predict on val
y_pred = pipeline.predict(X_val)
y_prob = pipeline.predict_proba(X_val)[:, 1]  # probabilities for positive class

## Model Performance (Optimized Threshold)

# Threshold Analysis
thresholds = np.arange(0, 1.005, 0.01)  
best_f1 = -1
best_threshold = 0
recall_scores, precision_scores, f1_scores = [], [], []

for threshold in thresholds:
    y_pred_thresh = (y_prob > threshold).astype(int)
    
    precision = precision_score(y_val, y_pred_thresh, zero_division=1)
    recall = recall_score(y_val, y_pred_thresh)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    recall_scores.append(recall)
    precision_scores.append(precision)
    f1_scores.append(f1)

    if f1 >= best_f1:
        best_f1 = f1
        best_threshold = threshold
        print(f"Threshold={threshold:.2f} Accuracy={accuracy_score(y_val, y_pred_thresh):.3f} Recall={recall:.2f} Precision={precision:.2f} F1={f1:.3f}")
print(f"Best Threshold: {best_threshold}")
plot_threshold_analysis(thresholds, recall_scores, precision_scores, f1_scores, best_threshold, filename="Model Performance/ThresholdAnalysis.png")

# Test predictions
y_pred = pipeline.predict(X_test) # get labels under default thresh
y_prob = pipeline.predict_proba(X_test)[:, 1] # get probabilities 
y_pred_best = (y_prob > best_threshold).astype(int) # assign new labels using best_thresh

# Confusion Matrix (Default Threshold)
plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix Default Threshold", "Model Performance/ConfusionMatrix_default.png")

# Classification Report (Default Threshold)
print("Classification Report - Default Threshold:\n", classification_report(y_test, y_pred))
 
# Confusion Matrix 
plot_confusion_matrix(y_test, y_pred_best, f"Confusion Matrix Optimal Threshold = {best_threshold:.2f}", "Model Performance/ConfusionMatrix_optimized.png")

# Classification Report 
print("Classification Report - Optimal Threshold:\n", classification_report(y_test, y_pred_best))

# ROC AUC
plot_roc_curve(y_test, y_prob, best_threshold, "Model Performance/ROC_Curve.png")

# Save results to a text file
with open("Model Performance/test_results.txt", "w") as f:
    f.write(f"Best Threshold: {best_threshold:.3f}\n")
    f.write(f"Test Accuracy: {accuracy_score(y_test, y_pred_best):.4f}\n")
    f.write(f"Test Precision: {precision_score(y_test, y_pred_best):.4f}\n")
    f.write(f"Test Recall: {recall_score(y_test, y_pred_best):.4f}\n")
    f.write(f"Test F1 Score: {f1_score(y_test, y_pred_best)}\n")