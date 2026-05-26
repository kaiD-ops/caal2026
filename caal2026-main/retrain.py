"""
retrain.py
----------
This script retrains the GalaxyClassifierS4D model with a few improvements
over the original training run that got us 64.2% - just under the 65% target.

What we changed and why:
  - Added a learning rate scheduler: instead of keeping the same LR the whole
    time, we let it slowly decay toward zero. This helps the model fine-tune
    in later epochs instead of bouncing around the optimal weights.
  - Added random flips as data augmentation: galaxies look the same upside-down
    or mirrored (unlike text or faces), so flipping is a free way to show the
    model more variety without needing more data.
  - Increased epochs from 15 to 25: the model was still improving at epoch 14,
    so cutting it at 15 was leaving performance on the table.
  - Saves the full training history inside the checkpoint so generate_plots.py
    can draw the training curves without needing to re-run anything.

How to run:
    python retrain.py

Output:
    model_params/galaxys4-29070.pth   (overwrites the old checkpoint)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import GalaxyClassifierS4D
from model.functions import export_model_parameters, load_data

# These are the same as the original training run except for EPOCHS.
# RNG_SEED is set to the team ERP ID for reproducibility - same seed means
# the train/val split will be identical every time you run this.

RNG_SEED   = 29070
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
COLORED    = False      # grayscale images (1 channel, not RGB)
BATCH_SIZE = 64
EPOCHS     = 25         # was 15 before, model was still improving at epoch 14
LR         = 0.0015     # same starting LR as before
CHECKPOINT = "model_params/galaxys4-29070.pth"

# fix randomness so results are reproducible
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

print(f"Device : {DEVICE}")
print(f"Epochs : {EPOCHS}  |  LR: {LR}  |  Batch size: {BATCH_SIZE}\n")



# Data loading
# load_data() pulls the GalaxyMNIST dataset from ./data (downloads if missing)
# We split 80% for training and 20% for validation, stratified so each class
# has roughly the same proportion in both splits.

print("Loading training data...")
X, y_onehot, y = load_data(root="./data", download=True, train=True, colored=COLORED)

x_train, x_val, y_train_oh, y_val_oh = train_test_split(
    X, y_onehot,
    test_size=0.2,
    random_state=RNG_SEED,
    stratify=y   # makes sure each class is represented fairly in both splits
)


# Data augmentation
# Galaxy images have no "right way up" - a spiral galaxy rotated 180 degrees
# is still a spiral galaxy. So randomly flipping images horizontally or
# vertically is a safe way to artificially double/quadruple the training data.
# We only augment the training set, never the validation set.


def augment(x):
    # 50% chance of flipping left-right
    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=[-1])
    # 50% chance of flipping top-bottom
    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=[-2])
    return x


class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps a tensor dataset and applies augmentation on the fly."""
    def __init__(self, x, y, augment_fn=None):
        self.x = x
        self.y = y
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment_fn:
            x = self.augment_fn(x)
        return x, self.y[idx]


# training loader uses augmentation; validation loader does not
train_loader = DataLoader(
    AugmentedDataset(x_train, y_train_oh, augment),
    batch_size=BATCH_SIZE,
    shuffle=True   # shuffle every epoch so the model sees data in different order
)
val_loader = DataLoader(
    TensorDataset(x_val, y_val_oh),
    batch_size=BATCH_SIZE
)

print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}\n")



# Model, loss, optimizer, scheduler
model = GalaxyClassifierS4D(colored=COLORED).to(DEVICE)

# CrossEntropyLoss expects raw logits (not softmax), so we use return_logits=True
loss_fn = nn.CrossEntropyLoss()

# Adam is a solid default optimizer - adapts learning rate per parameter
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Cosine annealing smoothly reduces the LR from its starting value down to
# eta_min over the course of all epochs. This avoids the model overshooting
# good weights in later epochs when it should be making small adjustments.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,   # decay over the full training run
    eta_min=1e-5    # never go below this LR
)


# Training loop
# We track loss and val accuracy every epoch. We also save the best model
# weights seen so far - this way even if the model degrades slightly in the
# last few epochs, we keep the best version.
history    = {"loss": [], "val_accuracy": []}
best_val   = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):

    # --- Training phase ---
    model.train()   # tells dropout/batchnorm layers to behave in training mode
    running_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()               # clear gradients from last step
        logits = model(xb, return_logits=True)
        loss   = loss_fn(logits, yb)
        loss.backward()                     # compute gradients
        optimizer.step()                    # update weights
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    scheduler.step()   # decay the learning rate for next epoch

    # --- Validation phase ---
    model.eval()    # tells layers to behave in inference mode
    correct = total = 0

    with torch.no_grad():   # no need to track gradients during evaluation
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds  = model(xb, return_logits=True).argmax(dim=1)
            labels = yb.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += len(labels)

    val_acc = correct / total
    history["loss"].append(avg_loss)
    history["val_accuracy"].append(val_acc)

    # keep a copy of the weights whenever we hit a new best val accuracy
    if val_acc > best_val:
        best_val   = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(
        f"Epoch {epoch:2d}/{EPOCHS}  "
        f"loss: {avg_loss:.4f}  "
        f"val_acc: {val_acc:.4f}"
        + ("  <- best" if val_acc == best_val else "")
    )


# Save checkpoint
# We save the best weights (not necessarily the last epoch), plus the full
# training history so generate_plots.py can draw the curves later.
os.makedirs("model_params", exist_ok=True)

torch.save({
    "model_state_dict" : best_state,
    "history"          : history,
    "best_val_acc"     : best_val,
}, CHECKPOINT)

print(f"\nCheckpoint saved to: {CHECKPOINT}")
print(f"Best validation accuracy: {best_val * 100:.2f}%")

# also export weights in the flat binary format the C implementation needs
print("\nExporting weights for C implementation...")
model.load_state_dict(best_state)
export_model_parameters(model, "model_params")
print("Done. Now run:  python generate_plots.py")
