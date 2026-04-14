"""
Generates all required M1 visualisation images from the saved model checkpoint.


Saves to images/:
    training_curves.png      - loss + val accuracy over epochs
    confusion_matrix.png     - normalised confusion matrix heatmap
    per_class_metrics.png    - precision, recall, F1 per class
    sample_predictions.png   - 3x3 grid of test predictions
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from model import GalaxyClassifierS4D
from model.functions import load_data

# Settings - make sure these match what you used during training
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "model_params/galaxys4-29070.pth"
COLORED    = False
BATCH_SIZE = 64
CLASSES    = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]
RNG_SEED   = 29070
os.makedirs("images", exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Loading checkpoint: {CHECKPOINT}")

# Load the trained model from checkpoint
# The checkpoint contains the model weights and training history

model = GalaxyClassifierS4D(colored=COLORED).to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

# handle both old format (raw state dict) and new format (dict with history)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    history = checkpoint.get("history", None)
else:
    model.load_state_dict(checkpoint)
    history = None

model.eval()
print("Model loaded.\n")

# Load test data and run inference
# We run the whole test set through the model to get predictions

print("Loading test data...")
X_test, y_test_onehot, y_test = load_data(
    root="./data", download=True, train=False, colored=COLORED
)
test_loader = DataLoader(TensorDataset(X_test, y_test_onehot), batch_size=BATCH_SIZE)

all_preds, all_probs, all_labels = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb     = xb.to(DEVICE)
        probs  = model(xb, return_logits=False)
        preds  = probs.argmax(dim=1)
        labels = yb.argmax(dim=1)
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_probs  = torch.cat(all_probs)
all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

accuracy = (all_preds == all_labels).mean()
print(f"Test Accuracy: {accuracy*100:.2f}%\n")

# Plot 1: Training curves
# Shows how loss and validation accuracy changed over epochs

if history:
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="#C44E52")
    ax1.plot(epochs, history["loss"], color="#C44E52", linewidth=2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="#C44E52")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Accuracy", color="#4C72B0")
    ax2.plot(epochs, history["val_accuracy"], color="#4C72B0", linewidth=2,
             linestyle="--", label="Val Accuracy")
    ax2.axhline(y=0.65, color="gray", linestyle=":", linewidth=1.2, label="65% target")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    plt.title("Training Loss and Validation Accuracy")
    fig.tight_layout()
    plt.savefig("images/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: images/training_curves.png")
else:
    print("No training history found in checkpoint - skipping training_curves.png")

# Plot 2: Confusion matrix
# Rows = true class, columns = predicted class, normalised by true class
# A perfect model would have 1.0 on the diagonal and 0.0 everywhere else

cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("Confusion Matrix (Normalised by True Class)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/confusion_matrix.png")


# Plot 3: Per-class precision, recall and F1
# Precision = of everything predicted as class X, how many were actually X
# Recall    = of all actual class X samples, how many did we catch
# F1        = harmonic mean of precision and recall

precision = precision_score(all_labels, all_preds, average=None)
recall    = recall_score(all_labels, all_preds, average=None)
f1        = f1_score(all_labels, all_preds, average=None)

x     = np.arange(len(CLASSES))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, precision, width, label="Precision", color="#4C72B0")
b2 = ax.bar(x,         recall,    width, label="Recall",    color="#55A868")
b3 = ax.bar(x + width, f1,        width, label="F1-Score",  color="#C44E52")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Score")
ax.set_title("Per-Class Precision, Recall and F1-Score")
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

macro_p = precision.mean()
macro_r = recall.mean()
macro_f = f1.mean()
ax.text(0.98, 0.02,
        f"Macro avg - Precision: {macro_p:.2f}  Recall: {macro_r:.2f}  F1: {macro_f:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

plt.tight_layout()
plt.savefig("images/per_class_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/per_class_metrics.png")

# Plot 4: Sample predictions grid (3x3)
# Shows 6 correct predictions and 3 confident wrong ones
# Green title = correct, red title = wrong

np.random.seed(RNG_SEED)

correct_idx   = np.where(all_preds == all_labels)[0]
incorrect_idx = np.where(all_preds != all_labels)[0]

# pick the 3 most confidently wrong predictions - interesting to analyse
incorrect_conf = all_probs[incorrect_idx].max(dim=1).values.numpy()
top_incorrect  = incorrect_idx[np.argsort(-incorrect_conf)[:3]]

selected = np.concatenate([np.random.choice(correct_idx, 6, replace=False), top_incorrect])
np.random.shuffle(selected)
selected = selected[:9]

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")

for ax, idx in zip(axes.flat, selected):
    img     = X_test[idx].squeeze().numpy()
    true    = all_labels[idx]
    pred    = all_preds[idx]
    conf    = all_probs[idx][pred].item()
    correct = (true == pred)

    ax.imshow(img, cmap="magma")
    ax.axis("off")

    color = "#2ecc71" if correct else "#e74c3c"
    ax.set_title(
        f"True:  {CLASSES[true]}\nPred:  {CLASSES[pred]}\nConf: {conf:.1%}",
        fontsize=8, color=color, pad=4
    )

plt.tight_layout()
plt.savefig("images/sample_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: images/sample_predictions.png")

print(f"\nAll done! Test accuracy: {accuracy*100:.2f}%")
