# Accuracy, Confusion matrix, plots
# Evaluate the model accuracy, confusion matrix, and classification report.

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=[
                                   "NonViolence", "Violence"])

    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["NonViolence", "Violence"],
                yticklabels=["NonViolence", "Violence"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return acc
