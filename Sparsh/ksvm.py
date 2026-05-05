import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.datasets import fetch_covtype
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import time
import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())

class Nystroem32(Nystroem):
    def fit_transform(self, X, y=None):
        return super().fit_transform(X.astype(np.float32)).astype(np.float32)
    def transform(self, X):
        return super().transform(X.astype(np.float32)).astype(np.float32)

print("Loading data...")
X, y = fetch_covtype(return_X_y=True)
X = X.astype(np.float32)
print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
del X_temp
gc.collect()

print("\n── Phase 1: C search (30k subset, n_components=6000) ──")

X_sub, _, y_sub, _ = train_test_split(
    X, y, train_size=30_000, random_state=42, stratify=y
)
scaler_sub = StandardScaler()
X_sub_scaled = scaler_sub.fit_transform(X_sub).astype(np.float32)
del X_sub
gc.collect()

nystroem_sub = Nystroem32(kernel="rbf", n_components=2000, random_state=42)
X_sub_ny = nystroem_sub.fit_transform(X_sub_scaled)
del X_sub_scaled
gc.collect()

C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
c_results = []

print(f"{'C':>10} {'acc':>8} {'std':>7} {'time':>7}")
print("-" * 36)

for C in C_values:
    start = time.time()
    fold_scores = []
    for train_idx, val_idx in skf.split(X_sub_ny, y_sub):
        clf = LinearSVC(dual=False, C=C, max_iter=2000)
        clf.fit(X_sub_ny[train_idx], y_sub[train_idx])
        fold_scores.append(accuracy_score(y_sub[val_idx], clf.predict(X_sub_ny[val_idx])))
        del clf
        gc.collect()
    elapsed = time.time() - start
    mean_acc = np.mean(fold_scores)
    std_acc  = np.std(fold_scores)
    c_results.append((C, mean_acc, std_acc, elapsed))
    print(f"{C:>10.3f} {mean_acc:>8.4f} {std_acc:>7.4f} {elapsed:>6.1f}s")

best_C = max(c_results, key=lambda x: x[1])[0]
print(f"\nBest C: {best_C}")

del X_sub_ny
gc.collect()

print(f"\n── Phase 2: Full Training (dual=False, C={best_C}, n_components=6000) ──")

print("\n[1/4] Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled   = scaler.transform(X_val).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)
del X_train, X_val, X_test
gc.collect()

print("[2/4] Nyström transform...")
t = time.time()
nystroem = Nystroem32(kernel="rbf", n_components=2000, random_state=42)
X_train_ny = nystroem.fit_transform(X_train_scaled)
X_val_ny   = nystroem.transform(X_val_scaled)
X_test_ny  = nystroem.transform(X_test_scaled)
del X_train_scaled, X_val_scaled, X_test_scaled
gc.collect()
print(f"      Done in {time.time()-t:.1f}s")
print(f"      Train matrix: {X_train_ny.nbytes/1e9:.2f} GB")

print("[3/4] Training LinearSVC(dual=False, class_weight='balanced')...")
t = time.time()
clf = LinearSVC(dual=False, C=best_C, max_iter=2000, tol=1e-1,
                class_weight='balanced', verbose=1)
clf.fit(X_train_ny, y_train)
train_time = time.time() - t
print(f"      Done in {train_time:.1f}s")

print("[4/4] Evaluating...")
train_acc = accuracy_score(y_train, clf.predict(X_train_ny))
val_acc   = accuracy_score(y_val,   clf.predict(X_val_ny))
test_acc  = accuracy_score(y_test,  clf.predict(X_test_ny))

print(f"\n{'='*40}")
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Val  Accuracy  : {val_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Train-Val Gap  : {train_acc - val_acc:.4f}")
print(f"{'='*40}")
print("\nPer-class breakdown (test set):")
y_test_pred = clf.predict(X_test_ny)
print(classification_report(y_test, y_test_pred,
                             target_names=[f"Class {i}" for i in range(1, 8)]))

print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=[f"Class {i}" for i in range(1, 8)])

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(f"Confusion Matrix — Test Accuracy: {test_acc:.4f}  |  n_components=6000  |  C={best_C}",
             fontsize=12, pad=15)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved to confusion_matrix.png")