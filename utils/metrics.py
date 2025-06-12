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

    # Ensure confusion matrix handles both classes even if one is missing
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    report = classification_report(
        all_labels,
        all_preds,
        labels=[0, 1],
        target_names=["NonViolence", "Violence"],
        zero_division=0
    )

    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        print(
            f"Warning: Only class {list(unique_labels)[0]} present in validation set.")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["NonViolence", "Violence"],
                yticklabels=["NonViolence", "Violence"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("Confusion Matrix.png")
    plt.show()

    return acc
