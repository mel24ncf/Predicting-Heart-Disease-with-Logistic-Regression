# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Helper functions

## Confusion Matrix
def plot_confusion_matrix(y_test, y_pred, title, filename):
    """
    Plots and saves a confusion matrix

    Args:
    ---------------
    y_test (array): True labels
    y_pred (array): Predicted labels
    title (str) - Title of confusion matrix
    filename (str) - Output filename for the saved figure

    Returns:
    ---------------
    cm (array): Confusion matrix
    """
    # construct cm
    cm = confusion_matrix(y_test, y_pred)

    # plotting
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No CHD", "CHD"], yticklabels=["No CHD", "CHD"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

    return cm

## Thresholds
def plot_threshold_analysis(thresholds, recall_scores, precision_scores, f1_scores, best_threshold, filename="thresholds.png"):
    """
    Plots the Precision-Recall-F1 Curve and marks the best threshold.

    Args:
    ---------------
        thresholds (array): Array of threshold values.
        recall_scores (array): Corresponding recall scores.
        precision_scores (array): Corresponding precision scores.
        f1_scores (array): Corresponding F1 scores.
        best_threshold (float): The best threshold value found.
        filename (str): Name of the file to save the figure.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, recall_scores, label="Recall", color="blue")
    plt.plot(thresholds, precision_scores, label="Precision", color="green")
    plt.plot(thresholds, f1_scores, label="F1", color="red")
    plt.axvline(best_threshold, linestyle="dashed", color="black", label=f"Best Threshold ({best_threshold:.2f})")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Threshold Analysis")
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

## ROC AUC
def plot_roc_curve(y_test, y_probs, best_threshold, filename="roc_curve.png"):
    """
    Plots the ROC curve and marks the best threshold.

    Args:
        y_test (array): True labels.
        y_probs (array): Predicted probabilities.
        best_threshold (float): The best threshold value found.
        filename (str): Name of the file to save the figure.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="gray", label="Random Classifier")
    
    best_idx = np.argmin(np.abs(thresholds - best_threshold))
    plt.scatter(fpr[best_idx], tpr[best_idx], color="red", label=f"Best Threshold ({best_threshold:.2f})")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()