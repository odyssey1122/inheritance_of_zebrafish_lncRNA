import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    roc_curve, average_precision_score
)
from sklearn.pipeline import Pipeline

# GC MATCHING FUNCTION
def gc_match_subsample(df, n_bins=5):
    print("\nRunning GC-matched subsampling...")

    df["gc_bin"] = pd.qcut(df["gc_content"], q=n_bins,
                           labels=False, duplicates="drop")

    matched_indices = []

    for b in sorted(df["gc_bin"].dropna().unique()):
        bin_df = df[df["gc_bin"] == b]

        inh = bin_df[bin_df["label"] == "inherited"]
        non = bin_df[bin_df["label"] == "non_inherited"]

        if len(inh) == 0 or len(non) == 0:
            continue

        n = min(len(inh), len(non))

        inh_sample = inh.sample(n, random_state=42)
        non_sample = non.sample(n, random_state=42)

        matched_indices.extend(inh_sample.index)
        matched_indices.extend(non_sample.index)

    df_matched = df.loc[matched_indices].copy()

    print(f"After GC matching: {len(df_matched)} sequences")
    print(df_matched["label"].value_counts())

    return df_matched


# PATHS
CSV_PATH = "train_features.csv"
OUT_PLOT = "kill_test_kmer2to5.png"

# LOAD DATA
df = pd.read_csv(CSV_PATH)
df = gc_match_subsample(df)

# GC BINNING FOR STRATIFICATION
print("\nCreating GC bins for matching...")

df["gc_bin"] = pd.qcut(df["gc_content"], q=5,
                       labels=False, duplicates="drop")

strat_labels = df["label"].astype(str) + "_" + df["gc_bin"].astype(str)

print("GC bin distribution:")
print(df["gc_bin"].value_counts().sort_index())

# FEATURE SELECTION: ONLY k for 2–5 
feature_cols = [
    col for col in df.columns
    if any(col.startswith(f"kmer{k}_") for k in range(2, 6))
]

print(f"\nUsing {len(feature_cols)} k-mer features (k=2–5)")

X = df[feature_cols].values.astype(float)
y = (df["label"] == "inherited").astype(int).values

# FEATURE IMPORTANCE USING RF
print("\nComputing RF feature importance...")

rf_full = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

rf_full.fit(X, y)

importances = rf_full.feature_importances_

feat_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 20 RF Features:")
print(feat_df.head(20))

feat_df.to_csv("rf_feature_importance_kmer2to5.csv", index=False)

print(f"Dataset: {X.shape[0]} sequences x {X.shape[1]} features")

# MODEL
models = {
    "LR": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, max_iter=1000,
            solver="lbfgs", random_state=42))
    ]),
    "RF": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    ),
}

# CROSS-VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    aucs, aps = [], []
    cms = np.zeros((2, 2), dtype=int)
    all_probs = np.zeros(len(y))
    all_true = np.zeros(len(y))
    roc_data = []

    for tr, te in cv.split(X, strat_labels):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_te, probs))
        aps.append(average_precision_score(y_te, probs))
        cms += confusion_matrix(y_te, preds)

        all_probs[te] = probs
        all_true[te] = y_te

        fpr, tpr, _ = roc_curve(y_te, probs)
        roc_data.append((fpr, tpr, aucs[-1]))

    tn, fp, fn, tp = cms.ravel()

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"AP:  {np.mean(aps):.3f} ± {np.std(aps):.3f}")
    print(f"Confusion matrix:\n{cms}")

    results[name] = {
        "aucs": aucs,
        "aps": aps,
        "cm": cms,
        "roc_data": roc_data
    }

# PERMUTATION TEST
print("\nRunning permutation test (RF)...")

null_aucs = []
rf = RandomForestClassifier(n_estimators=100,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=42)

for i in range(100):
    y_shuf = y.copy()
    np.random.shuffle(y_shuf)

    fold_aucs = []
    for tr, te in cv.split(X, y_shuf):
        rf.fit(X[tr], y_shuf[tr])
        probs = rf.predict_proba(X[te])[:, 1]
        fold_aucs.append(roc_auc_score(y_shuf[te], probs))

    null_aucs.append(np.mean(fold_aucs))

real_auc = np.mean(results["RF"]["aucs"])
pval = np.mean(np.array(null_aucs) >= real_auc)

print(f"RF real AUC: {real_auc:.3f}")
print(f"Null AUC: {np.mean(null_aucs):.3f}")
print(f"p-value: {pval:.3f}")

# SAVE PLOT
plt.hist(null_aucs, bins=20)
plt.axvline(real_auc)
plt.title("Permutation Test")
plt.savefig(OUT_PLOT, dpi=300)
plt.show()

print(f"\nSaved: {OUT_PLOT}")
