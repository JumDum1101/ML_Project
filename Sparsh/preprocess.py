from pathlib import Path
import pickle

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 0-9   : 10 continuous features
# 10-13 : 4 binary cols, one-hot for wilderness area
# 14-53 : 40 binary cols, one-hot for soil type
CONT_IDX = slice(0, 10)
WILD_IDX = slice(10, 14)
SOIL_IDX = slice(14, 54)

CONT_NAMES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

OUT_DIR = Path("./data")
SEED = 0

# one hot cols to id
def onehot_to_ids(block: np.ndarray, name: str) -> np.ndarray:
    row_sums = block.sum(axis=1)
    if not np.all(row_sums == 1):
        bad = np.sum(row_sums != 1)
        raise ValueError(
            f"{name}: {bad} rows do not have exactly one active column "
            f"(sums range {row_sums.min()}..{row_sums.max()})"
        )
    return block.argmax(axis=1).astype(np.int64)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Fetching Covertype (first run downloads ~11 MB)...")
    ds = fetch_covtype()
    X, y = ds.data.astype(np.float32), ds.target.astype(np.int64)
    y = y - 1  
    print(f"  X shape: {X.shape}   y shape: {y.shape}")

    print("\nClass distribution (label: count, pct):")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"  class {c}: {n:>7d}  ({100 * n / len(y):5.2f}%)")

    X_cont = X[:, CONT_IDX]
    X_wild = onehot_to_ids(X[:, WILD_IDX], "wilderness")
    X_soil = onehot_to_ids(X[:, SOIL_IDX], "soil")
    X_cat = np.stack([X_wild, X_soil], axis=1)

    print(f"\nContinuous features: {X_cont.shape}, dtype {X_cont.dtype}")
    print(f"  ranges: min={X_cont.min(axis=0)}")
    print(f"          max={X_cont.max(axis=0)}")
    print(f"Categorical features: {X_cat.shape}")
    print(f"  wilderness cardinality: {X_wild.max() + 1}")
    print(f"  soil cardinality:       {X_soil.max() + 1}")

    X_cont_tr, X_cont_tmp, X_cat_tr, X_cat_tmp, y_tr, y_tmp = train_test_split(
        X_cont, X_cat, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_cont_val, X_cont_te, X_cat_val, X_cat_te, y_val, y_te = train_test_split(
        X_cont_tmp, X_cat_tmp, y_tmp,
        test_size=2 / 3, stratify=y_tmp, random_state=SEED,
    )

    print(f"\nSplit sizes:  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}")

    print("\nClass proportions per split:")
    print(f"  {'cls':>3}  {'train':>7}  {'val':>7}  {'test':>7}")
    for c in classes:
        p_tr = (y_tr == c).mean()
        p_val = (y_val == c).mean()
        p_te = (y_te == c).mean()
        print(f"  {c:>3}  {p_tr:>7.4f}  {p_val:>7.4f}  {p_te:>7.4f}")

    scaler = StandardScaler()
    X_cont_tr = scaler.fit_transform(X_cont_tr).astype(np.float32)
    X_cont_val = scaler.transform(X_cont_val).astype(np.float32)
    X_cont_te = scaler.transform(X_cont_te).astype(np.float32)

    print(f"\nAfter scaling (train): mean={X_cont_tr.mean():.3e}  "
          f"std={X_cont_tr.std():.3e}")

    np.savez(
        OUT_DIR / "covtype_splits.npz",
        X_cont_train=X_cont_tr, X_cat_train=X_cat_tr, y_train=y_tr,
        X_cont_val=X_cont_val, X_cat_val=X_cat_val, y_val=y_val,
        X_cont_test=X_cont_te, X_cat_test=X_cat_te, y_test=y_te,
    )
    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    meta = {
        "cont_names": CONT_NAMES,
        "cat_names": ["wilderness_area", "soil_type"],
        "cat_cardinalities": [int(X_wild.max() + 1), int(X_soil.max() + 1)],
        "n_classes": int(classes.max() + 1),
        "seed": SEED,
    }
    with open(OUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved to {OUT_DIR.resolve()}/")
    print("  covtype_splits.npz, scaler.pkl, meta.pkl")


main()