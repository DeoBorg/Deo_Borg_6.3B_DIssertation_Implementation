"""
  - SMOTE oversampling to balance minority classes
  - Aggressive class weights (squared inverse frequency)
  - Higher dropout for regularisation
  - More epochs with patience-based early stopping
  - Per-fold best model saved based on macro F1 not accuracy

Usage:
    pip install imbalanced-learn
    python train_lstm.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import sys

# Install imbalanced-learn if not present
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Installing imbalanced-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "imbalanced-learn", "--quiet"])
    from imblearn.over_sampling import SMOTE

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)
from collections import Counter

# ── CONFIG ───────────────────────────────────────────────────────────────────
X_PATH      = "dataset/sequences_X.npy"
Y_PATH      = "dataset/sequences_y.npy"
MODEL_OUT   = "models/lstm_best.pt"
RESULTS_OUT = "outputs/lstm_fold_results.csv"
CM_OUT      = "outputs/lstm_confusion_matrix.png"
CURVES_OUT  = "outputs/lstm_training_curves.png"

SEQUENCE_LEN = 5
FEATURE_DIM  = 9
NUM_CLASSES  = 4
HIDDEN_SIZE  = 128   # increased from 64
NUM_LAYERS   = 2
DROPOUT      = 0.4   # increased from 0.3
EPOCHS       = 100   # increased from 50
BATCH_SIZE   = 32
LR           = 5e-4  # slightly lower for stability
K_FOLDS      = 5
PATIENCE     = 15    # early stopping patience

LABEL_NAMES  = ['Focused', 'Chatting', 'Looking Away', 'Using Phone']
# ─────────────────────────────────────────────────────────────────────────────


class BehaviourLSTM(nn.Module):
    def __init__(self, input_size=FEATURE_DIM, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
                 dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


def compute_class_weights(y):
    """Squared inverse frequency — much more aggressive than linear."""
    counts  = Counter(y.tolist())
    total   = len(y)
    weights = []
    for i in range(NUM_CLASSES):
        count = counts.get(i, 1)
        # Squared inverse frequency gives much more weight to minority classes
        w = (total / (NUM_CLASSES * count)) ** 2
        weights.append(w)
    # Normalise so weights sum to NUM_CLASSES
    weights = np.array(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to oversample minority classes.
    SMOTE works on 2D arrays so we flatten sequences,
    apply SMOTE, then reshape back.
    """
    n_samples, seq_len, n_features = X_train.shape
    X_flat = X_train.reshape(n_samples, seq_len * n_features)

    counts = Counter(y_train.tolist())
    print(f"    Before SMOTE: {dict(sorted(counts.items()))}")

    # Set k_neighbors based on smallest class
    min_count = min(counts.values())
    k_neighbors = min(5, min_count - 1)

    if k_neighbors < 1:
        print(f"    Skipping SMOTE — not enough minority samples")
        return X_train, y_train

    smote = SMOTE(
        random_state=42,
        k_neighbors=k_neighbors,
        sampling_strategy='auto'   # oversample all minority to match majority
    )

    X_resampled, y_resampled = smote.fit_resample(X_flat, y_train)
    X_resampled = X_resampled.reshape(-1, seq_len, n_features)

    counts_after = Counter(y_resampled.tolist())
    print(f"    After SMOTE : {dict(sorted(counts_after.items()))}")

    return X_resampled.astype(np.float32), y_resampled


def train_one_fold(X_train, y_train, X_val, y_val, device, fold):
    """Train with SMOTE + aggressive weights + early stopping on macro F1."""

    # Apply SMOTE to training set only
    print(f"  Applying SMOTE...")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    train_ds = TensorDataset(
        torch.tensor(X_train_sm, dtype=torch.float32),
        torch.tensor(y_train_sm, dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = BehaviourLSTM().to(device)
    weights   = compute_class_weights(torch.tensor(y_train_sm)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS, eta_min=1e-5
    )

    best_macro_f1 = 0.0
    best_state    = None
    patience_ctr  = 0
    train_losses  = []
    val_accs      = []

    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds   = model(X_batch).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y_batch.numpy())

        val_acc   = accuracy_score(all_true, all_preds)
        macro_f1  = f1_score(all_true, all_preds, average='macro',
                             zero_division=0, labels=list(range(NUM_CLASSES)))
        val_accs.append(val_acc)

        # Early stopping on macro F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:03d}/{EPOCHS} — "
                  f"loss: {train_losses[-1]:.4f}  "
                  f"val_acc: {val_acc:.4f}  "
                  f"macro_f1: {macro_f1:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Final predictions with best model
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds   = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    final_acc      = accuracy_score(all_true, all_preds)
    final_macro_f1 = f1_score(all_true, all_preds, average='macro',
                               zero_division=0, labels=list(range(NUM_CLASSES)))

    return final_acc, best_macro_f1, all_preds, all_true, \
           train_losses, val_accs, model


def plot_training_curves(all_train_losses, all_val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (losses, accs) in enumerate(zip(all_train_losses, all_val_accs)):
        axes[0].plot(losses, label=f'Fold {i+1}', alpha=0.7)
        axes[1].plot(accs,   label=f'Fold {i+1}', alpha=0.7)
    axes[0].set_title('Training Loss per Fold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_title('Validation Accuracy per Fold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVES_OUT, dpi=150)
    plt.close()
    print(f"  Saved training curves → {CURVES_OUT}")


def plot_confusion_matrix(all_true, all_preds):
    cm      = confusion_matrix(all_true, all_preds,
                               labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    axes[1].set_title('Confusion Matrix (Normalised)')
    axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {CM_OUT}")


def main():
    print("\n=== Phase 6 v2: LSTM Training (SMOTE + Aggressive Weighting) ===\n")

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    print(f"  Loaded X: {X.shape}  y: {y.shape}")
    print(f"  Class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = (y == i).sum()
        print(f"    {name:<15}: {count} ({count/len(y)*100:.1f}%)")

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\n  Device: Apple MPS (GPU) ✅")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n  Device: CUDA GPU ✅")
    else:
        device = torch.device('cpu')
        print(f"\n  Device: CPU")

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results     = []
    all_train_losses = []
    all_val_accs     = []
    all_true_total   = []
    all_preds_total  = []
    best_macro_f1    = 0.0
    best_model       = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'─'*50}")
        print(f"  Fold {fold+1} / {K_FOLDS}")
        print(f"{'─'*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        val_acc, fold_macro_f1, preds, true, t_losses, v_accs, model = \
            train_one_fold(X_train, y_train, X_val, y_val, device, fold+1)

        report = classification_report(
            true, preds,
            labels=list(range(NUM_CLASSES)),
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0
        )

        print(f"\n  Fold {fold+1} Results:")
        print(f"    Val Accuracy : {val_acc:.4f}")
        print(f"    Macro F1     : {fold_macro_f1:.4f}")
        print(f"    Per-class F1:")
        for name in LABEL_NAMES:
            print(f"      {name:<15}: {report[name]['f1-score']:.4f}")

        fold_results.append({
            'fold':         fold + 1,
            'val_accuracy': val_acc,
            'macro_f1':     fold_macro_f1,
            **{f'f1_{n}': report[n]['f1-score'] for n in LABEL_NAMES}
        })

        all_train_losses.append(t_losses)
        all_val_accs.append(v_accs)
        all_true_total.extend(true)
        all_preds_total.extend(preds)

        if fold_macro_f1 > best_macro_f1:
            best_macro_f1 = fold_macro_f1
            best_model    = model

    results_df = pd.DataFrame(fold_results)

    print(f"\n{'='*50}")
    print(f"  K-Fold Summary ({K_FOLDS} folds)")
    print(f"{'='*50}")
    print(f"  Mean Accuracy : {results_df['val_accuracy'].mean():.4f} "
          f"± {results_df['val_accuracy'].std():.4f}")
    print(f"  Mean Macro F1 : {results_df['macro_f1'].mean():.4f} "
          f"± {results_df['macro_f1'].std():.4f}")
    print(f"\n  Per-class Mean F1:")
    for name in LABEL_NAMES:
        col  = f'f1_{name}'
        mean = results_df[col].mean()
        std  = results_df[col].std()
        print(f"    {name:<15}: {mean:.4f} ± {std:.4f}")

    results_df.to_csv(RESULTS_OUT, index=False)
    torch.save(best_model.state_dict(), MODEL_OUT)
    print(f"\n  Saved fold results → {RESULTS_OUT}")
    print(f"  Saved best model   → {MODEL_OUT}")

    plot_training_curves(all_train_losses, all_val_accs)
    plot_confusion_matrix(all_true_total, all_preds_total)

    print(f"\n  Overall Classification Report (all folds combined):")
    print(classification_report(
        all_true_total, all_preds_total,
        labels=list(range(NUM_CLASSES)),
        target_names=LABEL_NAMES,
        zero_division=0
    ))
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()