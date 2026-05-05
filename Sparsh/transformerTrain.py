# Tranformer code
# There's A LOT of explanations in report as well :)
# Check software usage for directions

from pathlib import Path
import pickle
import time
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt

from rtdl_revisiting_models import FTTransformer


# Saves data (including the best transformer itself)
DATA_DIR = Path("./data")
FIG_DIR = DATA_DIR / "figures"
SEED = 0

# Search budget
N_TRIALS = 15
SUBSAMPLE_N = 30_000 # for searching hyperparamters
EPOCHS_SEARCH = 12
PATIENCE = 3 # early-stopping if no increase in f1 for 3 epochs

# Final training with best found model
EPOCHS_FINAL = 30 # just stop here
PATIENCE_FINAL = 5 # a little more than when searching for best hyperparameters

BATCH_SIZE = 512
DEVICE = torch.device("cpu") # mps wasn't working for me, plus probably good if you have windows

# This is all related to the dataset. Should be pretty self explnatory. Splits are decided by preprocess.py
COVTYPE_CLASS_NAMES = [
    "1: Spruce/Fir",
    "2: Lodgepole Pine",
    "3: Ponderosa Pine",
    "4: Cottonwood/Willow",
    "5: Aspen",
    "6: Douglas-fir",
    "7: Krummholz",
]

class CovtypeDataset(Dataset):
    def __init__(self, X_cont: np.ndarray, X_cat: np.ndarray, y: np.ndarray):
        self.X_cont = torch.from_numpy(X_cont).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_cont[i], self.X_cat[i], self.y[i]


def load_splits():
    d = np.load(DATA_DIR / "covtype_splits.npz")
    with open(DATA_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    splits = {
        "train": CovtypeDataset(d["X_cont_train"], d["X_cat_train"], d["y_train"]),
        "val":   CovtypeDataset(d["X_cont_val"],   d["X_cat_val"],   d["y_val"]),
        "test":  CovtypeDataset(d["X_cont_test"],  d["X_cat_test"],  d["y_test"]),
    }
    return splits, meta


def stratified_subsample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n >= len(y):
        return np.arange(len(y))
    idx = np.arange(len(y))
    sub_idx, _ = train_test_split(
        idx, train_size=n, stratify=y, random_state=seed
    )
    return sub_idx


# all model stuff from here
def build_model(cfg, n_cont, cat_cards, n_classes):
    return FTTransformer(
        n_cont_features=n_cont,
        cat_cardinalities=cat_cards,
        d_out=n_classes,
        n_blocks=cfg["n_blocks"],
        d_block=cfg["d_block"],
        attention_n_heads=8,
        attention_dropout=cfg["attention_dropout"],
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=cfg["ffn_dropout"],
        residual_dropout=0.0,
    ).to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_n = 0.0, 0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for x_cont, x_cat, y in loader:
            x_cont = x_cont.to(DEVICE, non_blocking=True)
            x_cat = x_cat.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x_cont, x_cat)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "loss": total_loss / total_n,
        "acc": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
        "preds": preds,
        "targets": targets,
    }


def train_with_early_stopping(
    model, train_loader, val_loader, criterion, optimizer,
    max_epochs, patience, verbose=True, log_prefix="",
):
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer)
        va = run_epoch(model, val_loader, criterion, optimizer=None)
        dt = time.time() - t0

        improved = va["macro_f1"] > best_f1
        if improved:
            best_f1 = va["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if verbose:
            flag = " *" if improved else ""
            print(f"{log_prefix}epoch {epoch:>3d} | "
                  f"train loss {tr['loss']:.4f} f1 {tr['macro_f1']:.4f} | "
                  f"val loss {va['loss']:.4f} f1 {va['macro_f1']:.4f} | "
                  f"{dt:.1f}s{flag}")

        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"], "train_f1": tr["macro_f1"],
            "val_loss": va["loss"],   "val_f1": va["macro_f1"],
            "time": dt,
        })

        if epochs_since_improve >= patience:
            if verbose:
                print(f"{log_prefix}early stop at epoch {epoch} (best val f1 {best_f1:.4f})")
            break

    return best_f1, best_state, history, best_epoch


# Plots for training and confusion matrix
def plot_training_curves(history, best_epoch, out_path):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_f1   = [h["train_f1"]   for h in history]
    val_f1     = [h["val_f1"]     for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs, train_loss, marker="o", label="Train loss", color="#1f77b4")
    ax.plot(epochs, val_loss,   marker="s", label="Val loss",   color="#ff7f0e")
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
               label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-entropy loss", fontsize=11)
    ax.set_title("Training and validation loss", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    ax = axes[1]
    ax.plot(epochs, train_f1, marker="o", label="Train macro-F1", color="#1f77b4")
    ax.plot(epochs, val_f1,   marker="s", label="Val macro-F1",   color="#ff7f0e")
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.6,
               label=f"Best val (epoch {best_epoch})")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_title("Training and validation macro-F1", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    plt.suptitle("FT-Transformer: Final Training Run", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(targets, preds, class_names, out_path):
    """Plot the test-set confusion matrix as raw counts and row-normalized recall."""
    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n = len(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Raw counts
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_title("Confusion Matrix (counts)", fontsize=13, pad=10)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row-normalized (recall)
    ax = axes[1]
    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Confusion Matrix (row-normalized / recall)", fontsize=13, pad=10)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("FT-Transformer: Test Set Confusion Matrix", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Samples from a range of hyperparamters.
def sample_config(rng: np.random.Generator) -> dict:
    return {
        "lr":                 float(10 ** rng.uniform(-4.5, -3.0)),     # ~3e-5 .. 1e-3
        "weight_decay":       float(10 ** rng.uniform(-6, -4)),         # 1e-6 .. 1e-4
        "d_block":            int(rng.choice([96, 128, 192, 256])),
        "n_blocks":           int(rng.choice([2, 3, 4])),
        "attention_dropout":  float(rng.uniform(0.0, 0.3)),
        "ffn_dropout":        float(rng.uniform(0.0, 0.3)),
    }

# Actually run the search
def run_search(splits, meta, n_cont, cat_cards, n_classes):
    print(f"\n{'='*70}\nHyperparameter search: {N_TRIALS} trials on {SUBSAMPLE_N} stratified samples\n{'='*70}")

    y_train = splits["train"].y.numpy()
    sub_idx = stratified_subsample_indices(y_train, SUBSAMPLE_N, seed=SEED)
    train_sub = Subset(splits["train"], sub_idx)
    print(f"Subsampled train: {len(train_sub)} rows (from {len(splits['train'])})")

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(splits["val"], batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    rng = np.random.default_rng(SEED)
    criterion = nn.CrossEntropyLoss()
    trials = []

    for trial in range(1, N_TRIALS + 1):
        cfg = sample_config(rng)
        torch.manual_seed(SEED + trial)
        np.random.seed(SEED + trial)

        print(f"\n--- Trial {trial}/{N_TRIALS} ---")
        print(f"cfg: {cfg}")
        t0 = time.time()

        model = build_model(cfg, n_cont, cat_cards, n_classes)
        optimizer = torch.optim.AdamW(
            model.make_parameter_groups(),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )

        best_f1, _, history, _ = train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer,
            max_epochs=EPOCHS_SEARCH, patience=PATIENCE,
            verbose=True, log_prefix="  ",
        )
        elapsed = time.time() - t0
        trials.append({
            "trial": trial, "cfg": cfg, "best_val_f1": best_f1,
            "epochs_run": len(history), "time_sec": elapsed,
        })
        print(f"  -> trial {trial} done: best val f1 {best_f1:.4f}  ({elapsed:.0f}s)")

    trials.sort(key=lambda t: t["best_val_f1"], reverse=True)
    print(f"\n{'='*70}\nSearch results (top 5):\n{'='*70}")
    for t in trials[:5]:
        print(f"  val f1 {t['best_val_f1']:.4f}  trial {t['trial']:>2d}  cfg {t['cfg']}")

    return trials[0]["cfg"], trials


def run_final(cfg, splits, n_cont, cat_cards, n_classes):
    print(f"\n{'='*70}\nFinal training on full train set with best config\n{'='*70}")
    print(f"cfg: {cfg}")

    train_loader = DataLoader(splits["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(splits["val"], batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(splits["test"], batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = build_model(cfg, n_cont, cat_cards, n_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.make_parameter_groups(),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, history, best_epoch = train_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer,
        max_epochs=EPOCHS_FINAL, patience=PATIENCE_FINAL,
        verbose=True, log_prefix="",
    )

    print(f"\nBest val macro-F1: {best_f1:.4f}  (loading that checkpoint for test)")
    model.load_state_dict(best_state)
    te = run_epoch(model, test_loader, criterion, optimizer=None)

    print(f"\nTest results:")
    print(f"  loss     : {te['loss']:.4f}")
    print(f"  accuracy : {te['acc']:.4f}")
    print(f"  macro-F1 : {te['macro_f1']:.4f}")
    print("\nPer-class report:")
    print(classification_report(te["targets"], te["preds"], digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(te["targets"], te["preds"]))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    curves_path = FIG_DIR / "ft_training_curves.png"
    cm_path = FIG_DIR / "ft_confusion_matrix.png"
    plot_training_curves(history, best_epoch, curves_path)
    plot_confusion_matrix(te["targets"], te["preds"], COVTYPE_CLASS_NAMES, cm_path)
    print(f"\nSaved training curves to {curves_path}")
    print(f"Saved confusion matrix to {cm_path}")

    return model, best_state, te, history, best_epoch


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {DEVICE}")

    splits, meta = load_splits()
    n_cont = splits["train"].X_cont.shape[1]
    cat_cards = meta["cat_cardinalities"]
    n_classes = meta["n_classes"]
    print(f"n_cont={n_cont}  cat_cardinalities={cat_cards}  n_classes={n_classes}")

    best_cfg, trials = run_search(splits, meta, n_cont, cat_cards, n_classes)

    _, best_state, te, history, best_epoch = run_final(
        best_cfg, splits, n_cont, cat_cards, n_classes
    )

    out = DATA_DIR / "ft_transformer.pt"
    torch.save({
        "state_dict": best_state,
        "config": {**best_cfg, "n_cont": n_cont,
                   "cat_cardinalities": cat_cards, "n_classes": n_classes},
        "test_metrics": {k: te[k] for k in ("loss", "acc", "macro_f1")},
        "final_history": history,
        "best_epoch": best_epoch,
    }, out)
    print(f"\nSaved best model to {out}")

    with open(DATA_DIR / "search_trials.json", "w") as f:
        json.dump(trials, f, indent=2, default=float)
    print(f"Saved search log to {DATA_DIR / 'search_trials.json'}")


main()