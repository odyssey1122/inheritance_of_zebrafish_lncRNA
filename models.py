"""
 models.py
 train and evaluate all models

 Models:
   1. Logistic Regression (baseline - basic training model)
      - Feature vector input from train_master.csv
      - class_weight=balanced for class imbalance
      - 5-fold stratified cross-validation
      - Coefficients = feature importance

   2. Random Forest (300 trees)
      - Feature vector input from train_master.csv
      - class_weight=balanced for class imbalance
      - 5-fold stratified cross-validation
      - Feature importance scores

   3. 1D CNN
      - Raw sequence input (one-hot encoded)
      - Global Average Pooling + Global Max Pooling
        combined — no need for padding anymore since GPA and GMA handles variable length
      - Weighted loss for class imbalance
      - Saliency maps for interpretability

 Primary metrics:
   F1 Score + AUC — NOT raw accuracy (since data is imbalanced)

 Design principles:
   - Standalone island
   - crashes on missing inputs
   - logging + argparse
   - GPU support with CPU fallback
   - All seeds fixed for reproducibility

 Input:
   - features_output/train_master.csv
   - features_output/test_master.csv
   - split_output/inherited_train.fasta
   - split_output/non_inherited_train.fasta
   - split_output/inherited_test.fasta
   - split_output/non_inherited_test.fasta

 Output:
   - models_output/lr_results.txt
   - models_output/rf_results.txt
   - models_output/cnn_results.txt
   - models_output/saliency_maps.csv
   - models_output/gc_vs_confidence.csv
   - models_output/shuffle_control.txt
   - models_output/model_comparison.csv
"""

import csv
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
import ushuffle

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# One-hot encoding
NUCLEOTIDE_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "U": [0, 0, 0, 1],
    "T": [0, 0, 0, 1],
}


def set_seeds(seed=42):
    """
    Fix all random seeds for reproducibility. Must be called before any model training.

    Arguments:
      seed — random seed (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info(f"Random seeds fixed: {seed}")


def load_feature_matrix(csv_path):
    """
    Load train_master.csv or test_master.csv into numpy arrays.

    Memory note:
      This loads the full feature matrix into memory. For 4,000 sequences x 1,600 features

    Arguments:
      csv_path — path to master CSV file

    Returns:
      X      — numpy array shape (n_sequences, n_features)
      y      — numpy array shape (n_sequences,) int 0/1
      names  — list of sequence names
      features — list of feature column names
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {csv_path}\n"
            f"Run feature_selection.py first."
        )

    names    = []
    labels   = []
    rows     = []
    features = None

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        features = [col for col in reader.fieldnames
                    if col not in ("sequence_name", "label")]

        for row in reader:
            names.append(row["sequence_name"])
            label = row["label"].strip().lower()
            if label == "inherited":
                labels.append(1)
            elif label == "non_inherited":
                labels.append(0)
            else:
                raise ValueError(
                    f"Unknown label '{row['label']}' in {csv_path}."
                    f"Expected 'inherited' or 'non_inherited'."  
            )
            rows.append([
                float(row.get(feat, 0.0) or 0.0)
                for feat in features
            ])

    X = np.array(rows,   dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    log.info(f"  Loaded {X.shape[0]} sequences x "
             f"{X.shape[1]} features from {csv_path.name}")

    return X, y, names, features


def one_hot_encode(seq):
    """
    Convert an RNA sequence string to a one-hot tensor.

    Output shape: (4, sequence_length) — channels first for PyTorch Conv1d which expects (batch, channels, length). Unknown characters encoded as [0,0,0,0]

    Arguments:
      seq — RNA sequence string (cleaned, uppercase)

    Returns:
      torch.FloatTensor shape (4, len(seq))
    """
    encoded = []
    for nt in seq:
        encoded.append(NUCLEOTIDE_MAP.get(nt, [0, 0, 0, 0]))

    # encoded is shape (length, 4) — transpose to (4, length)
    tensor = torch.FloatTensor(encoded).transpose(0, 1)
    return tensor


class LncRNADataset(Dataset):
    """
    PyTorch Dataset for raw lncRNA sequences.

    Loads sequences from FASTA files and returns one-hot encoded tensors one at a time. Used by DataLoader for batched CNN training.

    Note on batching variable length sequences:
      Since sequences have different lengths and we use Global Average + Max Pooling, we process one sequence per batch (batch_size=1) during inference.
      During training we use a custom collate function to handle variable lengths.

    Arguments:
      fasta_label_pairs — list of (fasta_path, label_int) tuples
                          label: inherited=1, non_inherited=0
    """

    def __init__(self, fasta_label_pairs):
        self.sequences = []  # list of (name, one_hot_tensor, label)

        for fasta_path, label in fasta_label_pairs:
            fasta_path = Path(fasta_path)

            if not fasta_path.exists():
                raise FileNotFoundError(
                    f"FASTA file not found: {fasta_path}"
                )

            current_name = None
            current_seq  = []

            with open(fasta_path, "r") as f:
                for line in f:
                    line = line.strip()

                    if line.startswith(">"):
                        if current_name is not None:
                            seq    = "".join(current_seq).upper().replace("T", "U")
                            tensor = one_hot_encode(seq)
                            self.sequences.append(
                                (current_name, tensor, label)
                            )
                        current_name = line[1:].strip().split()[0]
                        current_seq  = []

                    elif line:
                        current_seq.append(line)

            # Last sequence
            if current_name is not None:
                seq    = "".join(current_seq).upper().replace("T", "U")
                tensor = one_hot_encode(seq)
                self.sequences.append((current_name, tensor, label))

        log.info(f"  Dataset loaded: {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        name, tensor, label = self.sequences[idx]
        return tensor, torch.FloatTensor([label]), name


def collate_variable_length(batch):
    """
    Custom collate function for variable length sequences.

    Since sequences have different lengths we cannot stack them into a single tensor directly. Instead we return them as a list and process each one individually in the forward pass.

    Arguments:
      batch — list of (tensor, label, name) tuples

    Returns:
      tensors — list of FloatTensors
      labels  — stacked label tensor
      names   — list of sequence names
    """
    tensors = [item[0] for item in batch]
    labels  = torch.stack([item[1] for item in batch])
    names   = [item[2] for item in batch]
    return tensors, labels, names


class MultiScaleCNN(nn.Module):
    """
    Multi-scale 1D CNN for lncRNA sequence classification.

    Three parallel branches with different kernel sizes:
      Branch 1 — kernel=8  : local patterns (short motifs)
      Branch 2 — kernel=32 : mid-range composition
      Branch 3 — kernel=64 : broad global context

    Each branch applies Conv1D -> ReLU -> GAP + GMP.
    GAP captures mean activation across the sequence.
    GMP captures the strongest activation anywhere.
    All branches concatenated -> dense classifier.

    Model learns which scale is informative from data; no prior assumption about which kernel dominates.

    Input:  (1, 4, sequence_length)
    Output: scalar logit
    """

    def __init__(self, filters=32, dropout=0.5):
        super(MultiScaleCNN, self).__init__()

        # One conv layer per branch — single layer keeps
        # parameter count low for small dataset (~2700 seqs)
        self.branch_8  = nn.Conv1d(4, filters, kernel_size=8,  padding=3)
        self.branch_32 = nn.Conv1d(4, filters, kernel_size=32, padding=15)
        self.branch_64 = nn.Conv1d(4, filters, kernel_size=64, padding=31)

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Each branch produces filters*2 values (GAP + GMP)
        # Three branches: filters * 2 * 3 = 192 with filters=32
        dense_input = filters * 2 * 3

        self.fc1 = nn.Linear(dense_input, 64)
        self.fc2 = nn.Linear(64, 1)

    def branch_forward(self, conv_layer, x):
        """
        Forward pass for a single branch.
        Applies conv -> ReLU -> GAP + GMP.
        Returns concatenated pooled vector of shape (1, filters*2).
        """
        out = self.relu(conv_layer(x))  # (1, filters, L')
        gap = out.mean(dim=2)           # (1, filters)
        gmp = out.max(dim=2).values     # (1, filters)
        return torch.cat([gap, gmp], dim=1)  # (1, filters*2)

    def forward(self, x):
        """
        Forward pass for a single sequence.
        Arguments:
          x — FloatTensor shape (1, 4, sequence_length)
        Returns:
          scalar logit
        """
        b8  = self.branch_forward(self.branch_8,  x)
        b32 = self.branch_forward(self.branch_32, x)
        b64 = self.branch_forward(self.branch_64, x)

        # Concatenate all branches: (1, 192)
        combined = torch.cat([b8, b32, b64], dim=1)

        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """
    Compute all evaluation metrics for a model.

    Primary metrics: F1 + AUC (not accuracy — imbalanced data)

    Arguments:
      y_true      — true labels (numpy array)
      y_pred      — predicted labels (numpy array)
      y_prob      — predicted probabilities (numpy array)
      model_name  — string for logging

    Returns:
      dictionary of metrics
    """
    f1        = f1_score(y_true, y_pred, average="macro")
    try:
        auc   = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc   = float("nan")
        log.warning("AUC could not be computed - single class in predictions")
    precision = precision_score(y_true, y_pred, average="weighted",
                                zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted",
                             zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    log.info(f"  {model_name} Results:")
    log.info(f"  F1 Score:  {f1:.4f}")
    log.info(f"  AUC:       {auc:.4f}")
    log.info(f"  Precision: {precision:.4f}")
    log.info(f"  Recall:    {recall:.4f}")
    log.info(f"  Confusion Matrix:")
    log.info(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    log.info(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        "model":     model_name,
        "f1":        f1,
        "auc":       auc,
        "precision": precision,
        "recall":    recall,
        "tn":        cm[0, 0],
        "fp":        cm[0, 1],
        "fn":        cm[1, 0],
        "tp":        cm[1, 1],
    }


def save_results(metrics, feature_importance, output_path):
    """
    Save model results to a text file.

    Arguments:
      metrics           — dictionary from evaluate_model
      feature_importance — list of (feature, score) tuples or None
      output_path       — path to write results
    """
    with open(output_path, "w") as f:
        f.write(f"MODEL: {metrics['model']}\n")
        f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
        f.write(f"AUC:       {metrics['auc']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TN={metrics['tn']}  FP={metrics['fp']}\n")
        f.write(f"  FN={metrics['fn']}  TP={metrics['tp']}\n\n")

        if feature_importance is not None:
            f.write("Top 20 Features by Importance:\n")
            f.write("-" * 40 + "\n")
            for feat, score in feature_importance[:20]:
                f.write(f"  {feat:<40} {score:.6f}\n")

    log.info(f"  Results saved to {output_path}")


def train_logistic_regression(X_train, y_train, X_test, y_test,
                               feature_names, output_path):
    """
    Train and evaluate Logistic Regression.

    Baseline model — interpretable coefficients show which features drive inherited vs non_inherited prediction.

    Arguments:
      X_train      — training feature matrix (numpy)
      y_train      — training labels (numpy)
      X_test       — test feature matrix (numpy)
      y_test       — test labels (numpy)
      feature_names — list of feature column names
      output_path  — where to save results

    Returns:
      metrics dictionary
    """
    log.info("Training Logistic Regression")

    # 5-fold cross-validation on train set
    log.info("  Running 5-fold stratified cross-validation...")
    skf      = StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=42)
    cv_f1s   = []
    cv_aucs  = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train)):

        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val   = X_train[val_idx]
        y_fold_val   = y_train[val_idx]
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_fold_train)
        X_fold_val = scaler.transform(X_fold_val)

        fold_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        )
        fold_model.fit(X_fold_train, y_fold_train)

        fold_pred = fold_model.predict(X_fold_val)
        fold_prob = fold_model.predict_proba(X_fold_val)[:, 1]

        cv_f1s.append(
            f1_score(y_fold_val, fold_pred, average="macro")
        )
        try:
            auc = roc_auc_score(y_fold_val, fold_prob)
        except ValueError:
            auc = float("nan")
        cv_aucs.append(auc)

        log.info(f"  Fold {fold+1}: "
                 f"F1={cv_f1s[-1]:.4f}  AUC={cv_aucs[-1]:.4f}")

    log.info(f"  CV F1:  {np.mean(cv_f1s):.4f} "
             f"± {np.std(cv_f1s):.4f}")
    log.info(f"  CV AUC: {np.mean(cv_aucs):.4f} "
             f"± {np.std(cv_aucs):.4f}")

    # Final model trained on full train set
    log.info("  Training final model on full train set...")
    
    final_scaler = StandardScaler()
    X_train = final_scaler.fit_transform(X_train)
    X_test = final_scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob,
                             "Logistic Regression")

    # Feature importance — top coefficients
    coefficients = model.coef_[0]
    importance   = sorted(
        zip(feature_names, coefficients),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Save results 
    save_results(metrics, importance, output_path)

    # Add CV scores to results file
    with open(output_path, "a") as f:
        f.write("\n5-Fold Cross-Validation (train set):\n")
        f.write(f"  F1:  {np.mean(cv_f1s):.4f} "
                f"± {np.std(cv_f1s):.4f}\n")
        f.write(f"  AUC: {np.mean(cv_aucs):.4f} "
                f"± {np.std(cv_aucs):.4f}\n")

    return metrics


def train_random_forest(X_train, y_train, X_test, y_test,
                        feature_names, output_path):
    """
    Train and evaluate Random Forest.

    300 trees — interpretable via feature importance scores. Most important features are biological signals the model found useful for inherited vs non inherited classification.

    Arguments:
      X_train       — training feature matrix (numpy)
      y_train       — training labels (numpy)
      X_test        — test feature matrix (numpy)
      y_test        — test labels (numpy)
      feature_names — list of feature column names
      output_path   — where to save results

    Returns:
      metrics dictionary
    """
    log.info("Training Random Forest")

    # 5-fold cross-validation on train set
    log.info("  Running 5-fold stratified cross-validation...")
    skf     = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=42)
    cv_f1s  = []
    cv_aucs = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train)):

        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val   = X_train[val_idx]
        y_fold_val   = y_train[val_idx]

        fold_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        fold_model.fit(X_fold_train, y_fold_train)

        fold_pred = fold_model.predict(X_fold_val)
        fold_prob = fold_model.predict_proba(X_fold_val)[:, 1]

        cv_f1s.append(
            f1_score(y_fold_val, fold_pred, average="macro")
        )
        try:
            auc = roc_auc_score(y_fold_val, fold_prob)
        except ValueError:
            auc = float("nan")
        cv_aucs.append(auc)

        log.info(f"  Fold {fold+1}: "
                 f"F1={cv_f1s[-1]:.4f}  AUC={cv_aucs[-1]:.4f}")

    log.info(f"  CV F1:  {np.mean(cv_f1s):.4f} "
             f"± {np.std(cv_f1s):.4f}")
    log.info(f"  CV AUC: {np.mean(cv_aucs):.4f} "
             f"± {np.std(cv_aucs):.4f}")

    # Final model trained on full train set
    log.info("  Training final model on full train set...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate on test set 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob,
                             "Random Forest")

    # Feature importance
    importances = model.feature_importances_
    importance  = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    # Save results
    save_results(metrics, importance, output_path)

    with open(output_path, "a") as f:
        f.write("\n5-Fold Cross-Validation (train set):\n")
        f.write(f"  F1:  {np.mean(cv_f1s):.4f} "
                f"± {np.std(cv_f1s):.4f}\n")
        f.write(f"  AUC: {np.mean(cv_aucs):.4f} "
                f"± {np.std(cv_aucs):.4f}\n")

    return metrics


def train_cnn(train_fasta_pairs, test_fasta_pairs,
              output_path, device,
              epochs=50, batch_size=32,
              learning_rate=0.001):
    """
    Train and evaluate 1D CNN on raw sequences.

    Architecture: Conv1D + Conv1D + GAP + GMP + Dense
    No padding — global pooling handles variable length.
    GAP + GMP combined for richer sequence representation.

    Arguments:
      train_fasta_pairs — list of (fasta_path, label_int)
      test_fasta_pairs  — list of (fasta_path, label_int)
      output_path       — where to save results
      epochs            — training epochs (default 50)
      batch_size        — batch size (default 32)
      learning_rate     — Adam lr (default 0.001)

    Returns:
      metrics dictionary
    """
    log.info("Training 1D CNN")
    log.info(f"  Device: {device}")

    # Load datasets
    log.info("  Loading train sequences")
    train_dataset = LncRNADataset(train_fasta_pairs)

    log.info("  Loading test sequences")
    test_dataset  = LncRNADataset(test_fasta_pairs)

    # Class imbalance weight
    train_labels  = [item[1].item() for item in train_dataset]
    n_inherited    = sum(train_labels)
    n_non_inherited     = len(train_labels) - n_inherited
    pos_weight    = torch.FloatTensor(
        [n_non_inherited / max(n_inherited, 1)]
    ).to(device)

    log.info(f"  Inherited: {int(n_inherited)} | "
             f"Non-Inherited: {int(n_non_inherited)}")
    log.info(f"  pos_weight: {pos_weight.item():.3f}")

    # DataLoaders
    # collate_variable_length handles the batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        collate_fn=collate_variable_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(42),
        collate_fn=collate_variable_length
    )

    # Model + loss + optimizer
    model     = MultiScaleCNN(filters=32, dropout=0.5).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    # Training loop with early stopping
    best_val_auc   = 0.0
    patience       = 5
    patience_count = 0
    best_state     = None

    # Split train into train/val (80/20)
    # further splits to the train set for validation
    # not test set - remains locked 
    # validation for "early stopping"
    n_train   = len(train_dataset)
    n_val     = max(1, int(n_train * 0.2))
    n_tr      = n_train - n_val
    tr_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        collate_fn=collate_variable_length
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(42),
        collate_fn=collate_variable_length
    )

    log.info(f"  Train: {n_tr} | Val: {n_val} | "
             f"Test: {len(test_dataset)}")
    log.info(f"  Epochs: {epochs} | "
             f"Early stopping patience: {patience}")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for tensors, labels, names in tr_loader:
            optimizer.zero_grad()

            # Process each sequence individually
            # (variable length — cannot stack)
            batch_loss = 0.0

            for seq_tensor, label in zip(tensors, labels):
                seq_tensor = seq_tensor.unsqueeze(0).to(device)
                label      = label.to(device)
                output     = model(seq_tensor)
                loss       = criterion(
                    output.view(-1), label.view(-1)
                ) / len(tensors)
                loss.backward()
                train_loss += loss.item()

            optimizer.step()
            
        train_loss /= len(tr_loader)

        # Validate
        # Validate
        model.eval()
        val_loss       = 0.0
        all_val_probs  = []
        all_val_labels = []

        with torch.no_grad():
            for tensors, labels, names in val_loader:
                batch_val_loss = 0.0
                for seq_tensor, label in zip(tensors, labels):
                    seq_tensor = seq_tensor.unsqueeze(0).to(device)
                    label      = label.to(device)
                    output     = model(seq_tensor)
                    loss       = criterion(
                        output.view(-1), label.view(-1)
                    )
                    batch_val_loss += loss.item()
                    all_val_probs.append(
                        torch.sigmoid(output).item()
                    )
                    all_val_labels.append(int(label.item()))
                val_loss += batch_val_loss / len(tensors)

        val_loss /= max(len(val_loader), 1)


        # Early stopping
        # Early stopping on AUC
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError:
            val_auc = 0.0

        log.info(f"  Epoch {epoch+1}/{epochs} — "
                 f"train_loss={train_loss:.4f}  "
                 f"val_loss={val_loss:.4f}  "
                 f"val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_state     = {k: v.clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("  Loaded best model weights")

    # Evaluate on test set
    model.eval()
    all_preds  = []
    all_probs  = []
    all_labels = []
    all_names  = []

    with torch.no_grad():
        for tensors, labels, names in test_loader:
            for seq_tensor, label, name in zip(
                    tensors, labels, names):
                seq_tensor = seq_tensor.unsqueeze(0).to(device)
                output     = model(seq_tensor)
                prob       = torch.sigmoid(output).item()
                pred       = 1 if prob >= 0.5 else 0

                all_probs.append(prob)
                all_preds.append(pred)
                all_labels.append(int(label.item()))
                all_names.append(name)

    y_true   = np.array(all_labels)
    y_pred   = np.array(all_preds)
    y_prob   = np.array(all_probs)
    metrics  = evaluate_model(y_true, y_pred, y_prob, "CNN")
    save_results(metrics, None, output_path)

    return model, metrics, all_names, all_probs, all_labels


def compute_saliency_maps(model, test_fasta_pairs, device,
                           output_path, top_n=5):
    """
    Compute saliency maps for top N most confident predictions per class.

    Saliency = gradient of output with respect to input.
    High gradient at a position means that nucleotide strongly influenced the prediction.

    If saliency peaks align with known motifs, CNN focused on localized sequences that may map to biologically relevant motifs.

    Arguments:
      model           — trained LncRNACNN
      test_fasta_pairs — list of (fasta_path, label_int)
      device          — torch.device
      output_path     — where to save saliency CSV
      top_n           — sequences per class to analyse

    Output CSV columns:
      name, true_label, predicted_label, confidence,
      position, saliency_score
    """
    log.info("Computing saliency maps...")

    model.eval()
    test_dataset = LncRNADataset(test_fasta_pairs)

    # Collect predictions and confidences
    results = []

    for seq_tensor, label, name in test_dataset:
        seq_tensor_in = seq_tensor.unsqueeze(0).to(device)
        seq_tensor_in.requires_grad_(False)

        with torch.no_grad():
            output = model(seq_tensor_in)
            prob   = torch.sigmoid(output).item()
            pred   = 1 if prob >= 0.5 else 0

        results.append({
            "name":       name,
            "true_label": int(label.item()),
            "pred_label": pred,
            "confidence": prob,
            "tensor":     seq_tensor,
        })

    # Select top N most confident per class
    inherited_results = sorted(
        [r for r in results if r["pred_label"] == 1],
        key=lambda x: x["confidence"],
        reverse=True
    )[:top_n]

    non_inherited_results = sorted(
        [r for r in results if r["pred_label"] == 0],
        key=lambda x: 1 - x["confidence"],
        reverse=True
    )[:top_n]

    selected = inherited_results + non_inherited_results
    log.info(f"  Computing saliency for {len(selected)} sequences")

    # Compute saliency for each selected sequence 
    saliency_rows = []

    for result in selected:
        seq_tensor = result["tensor"].unsqueeze(0).to(device)
        seq_tensor.requires_grad_(True)

        # Forward pass
        output = model(seq_tensor)
        prob   = torch.sigmoid(output)


        # Backward pass — gradient of output w.r.t. input
        if seq_tensor.grad is not None:
            seq_tensor.grad.zero_()
        model.zero_grad()
        if prob.item() >= 0.5:
            output.backward()
        else:
            (-output).backward()

        # Saliency = max absolute gradient across nucleotide
        # channels at each position
        saliency = seq_tensor.grad.abs().max(dim=1).values
        saliency = saliency.squeeze().cpu().numpy()

        # Normalize saliency to 0-1 range
        saliency_raw = saliency.copy()
        saliency_min = saliency.min()
        saliency_max = saliency.max()
        if saliency_max > saliency_min:
            saliency_norm = (saliency - saliency_min) / (
                saliency_max - saliency_min
            )
        else:
            saliency_norm = saliency.copy()

        # Write one row per position
        for pos, score in enumerate(saliency):
            saliency_rows.append({
                "name":          result["name"],
                "true_label":    result["true_label"],
                "pred_label":    result["pred_label"],
                "confidence":    result["confidence"],
                "position":      pos,
                "saliency_raw": float(saliency_raw[pos]),
                "saliency_normalized": float(saliency_norm[pos]),
            })

    # Save to CSV
    if len(saliency_rows) > 0:
        header = ["name", "true_label", "pred_label",
                  "confidence", "position", 
                  "saliency_raw", "saliency_normalized"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(saliency_rows)

        log.info(f"  Saliency maps saved to {output_path}")
    else:
        log.warning("  No saliency maps computed — check predictions")
        
        
def compute_gc_vs_confidence(test_fasta_pairs, all_names,
                              all_probs, all_labels, output_path):
    """
    Compute GC content vs prediction confidence for each test sequence.

    If GC content correlates strongly with confidence the model may be a GC detector not a biology detector. This plot is the confounder defense.

    Arguments:
      test_fasta_pairs — list of (fasta_path, label_int)
      all_names        — sequence names from CNN evaluation
      all_probs        — predicted probabilities from CNN
      all_labels       — true labels from CNN evaluation
      output_path      — where to save CSV
    """
    log.info("Computing GC content vs prediction confidence...")

    # Build lookup from name to prob and label
    prob_lookup  = dict(zip(all_names, all_probs))
    label_lookup = dict(zip(all_names, all_labels))

    rows = []

    for fasta_path, label_int in test_fasta_pairs:
        fasta_path   = Path(fasta_path)
        current_name = None
        current_seq  = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()

                if line.startswith(">"):
                    if current_name is not None:
                        seq = "".join(current_seq).upper().replace("T", "U")
                        n   = len(seq)
                        gc  = (seq.count("G") + seq.count("C")) / n \
                              if n > 0 else 0.0

                        if current_name in prob_lookup:
                            rows.append({
                                "name":       current_name,
                                "true_label": label_lookup.get(
                                    current_name, label_int),
                                "confidence": prob_lookup[current_name],
                                "gc_content": gc,
                            })

                    current_name = line[1:].strip().split()[0]
                    current_seq  = []

                elif line:
                    current_seq.append(line)

        # Last sequence
        if current_name is not None:
            seq = "".join(current_seq).upper().replace("T", "U")
            n   = len(seq)
            gc  = (seq.count("G") + seq.count("C")) / n \
                  if n > 0 else 0.0

            if current_name in prob_lookup:
                rows.append({
                    "name":       current_name,
                    "true_label": label_lookup.get(
                        current_name, label_int),
                    "confidence": prob_lookup[current_name],
                    "gc_content": gc,
                })

    if len(rows) > 1:
        gcs   = np.array([r["gc_content"] for r in rows])
        confs = np.array([r["confidence"] for r in rows])

        # np.corrcoef returns 2x2 matrix — [0,1] is the cross-correlation
        pearson = np.corrcoef(gcs, confs)[0, 1]
        log.info(f"  GC vs confidence Pearson r = {pearson:.4f}")

        if abs(pearson) > 0.5:
            log.warning(
                f"  High correlation ({pearson:.4f}) — "
                f"model may be learning GC bias"
            )
        else:
            log.info(
                f"  Low correlation — "
                f"model likely learning biology not GC bias"
            )

    # Save to CSV
    if len(rows) > 0:
        header = ["name", "true_label", "confidence", "gc_content"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"  GC vs confidence saved to {output_path}")
    else:
        log.warning("  No GC vs confidence data computed")


def dinucleotide_shuffle(seq):
    """
    Shuffle sequence preserving dinucleotide frequencies.
    Uses ushuffle library implementing the
    Altschul-Erikson algorithm.
    """
    if len(seq) < 3:
        return seq

    shuffled = ushuffle.shuffle(seq, len(seq), 2)
    return shuffled

def run_shuffle_control(train_fasta_pairs, test_fasta_pairs,
                        device, output_path,
                        epochs=20):
    """
    Retrain CNN on dinucleotide-shuffled sequences.

    If performance on shuffled sequences is similar to real sequences → model learned GC/dinucleotide bias but if performance drops significantly → model learned real biological motifs.

    Arguments:
      train_fasta_pairs — list of (fasta_path, label_int)
      test_fasta_pairs  — list of (fasta_path, label_int)
      device            — torch.device
      output_path       — where to save results
      epochs            — training epochs (default 20)
    """
    log.info("Running dinucleotide shuffle control...")
    log.info("  Shuffling sequences preserving dinucleotide freq")

    # Write shuffled FASTA files to temp location
    import tempfile

    shuffled_train_pairs = []
    shuffled_test_pairs  = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for fasta_path, label in train_fasta_pairs:
            out_path = tmp_path / f"shuffled_train_{label}.fasta"
            with open(fasta_path, "r") as fin, \
                 open(out_path, "w") as fout:
                current_name = None
                current_seq  = []

                for line in fin:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_name is not None:
                            seq      = "".join(current_seq).upper().replace("T", "U")
                            shuffled = dinucleotide_shuffle(seq)
                            fout.write(f">{current_name}\n")
                            fout.write(f"{shuffled}\n")
                        current_name = line[1:].strip().split()[0]
                        current_seq  = []
                    elif line:
                        current_seq.append(line)

                if current_name is not None:
                    seq      = "".join(current_seq).upper().replace("T", "U")
                    shuffled = dinucleotide_shuffle(seq)
                    fout.write(f">{current_name}\n")
                    fout.write(f"{shuffled}\n")

            shuffled_train_pairs.append((str(out_path), label))

        for fasta_path, label in test_fasta_pairs:
            out_path = tmp_path / f"shuffled_test_{label}.fasta"
            with open(fasta_path, "r") as fin, \
                 open(out_path, "w") as fout:
                current_name = None
                current_seq  = []

                for line in fin:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_name is not None:
                            seq      = "".join(current_seq).upper().replace("T", "U")
                            shuffled = dinucleotide_shuffle(seq)
                            fout.write(f">{current_name}\n")
                            fout.write(f"{shuffled}\n")
                        current_name = line[1:].strip().split()[0]
                        current_seq  = []
                    elif line:
                        current_seq.append(line)

                if current_name is not None:
                    seq      = "".join(current_seq).upper().replace("T", "U")
                    shuffled = dinucleotide_shuffle(seq)
                    fout.write(f">{current_name}\n")
                    fout.write(f"{shuffled}\n")

            shuffled_test_pairs.append((str(out_path), label))

        # Train CNN on shuffled sequences
        log.info("  Training CNN on shuffled sequences...")
        tmp_output = Path(output_path).with_suffix(".tmp.txt")
        _, shuffle_metrics, _, _, _ = train_cnn(
            shuffled_train_pairs,
            shuffled_test_pairs,
            tmp_output,  # separate temp file
            device,
            epochs=epochs
        )

    # Save comparison
    with open(output_path, "w") as f:
        f.write("DINUCLEOTIDE SHUFFLE CONTROL\n")
        f.write("If shuffled performance ≈ real performance:\n")
        f.write("  - Model learned GC/dinucleotide bias\n\n")
        f.write("If shuffled performance << real performance:\n")
        f.write("  - Model learned real biological motifs\n\n")
        f.write(f"Shuffled CNN F1:  {shuffle_metrics['f1']:.4f}\n")
        f.write(f"Shuffled CNN AUC: {shuffle_metrics['auc']:.4f}\n")

    log.info(f"  Shuffle control results saved to {output_path}")
    return shuffle_metrics


def analyze_errors(test_fasta_pairs, all_names, all_preds,
                   all_probs, all_labels, feature_matrix_path,
                   output_dir):
    """
    Error analysis on test set predictions.

    Outputs two files:
      (A) error_analysis.csv — one row per sequence with error type (TP/FP/TN/FN), confidence, GC, length
      (B) error_summary.csv — group-level means per error type plus confidence separation (correct vs incorrect)

    Arguments:
      test_fasta_pairs    — list of (fasta_path, label_int)
      all_names           — sequence names from model evaluation
      all_preds           — predicted labels (list of int)
      all_probs           — predicted probabilities (list of float)
      all_labels          — true labels (list of int)
      feature_matrix_path — path to test_master.csv for feature lookup
      output_dir          — Path to models_output/
    """
    log.info("Running error analysis...")

    # Assign error type per sequence
    def error_type(true, pred):
        if true == 1 and pred == 1:
            return "TP"
        elif true == 0 and pred == 0:
            return "TN"
        elif true == 0 and pred == 1:
            return "FP"
        else:
            return "FN"

    # Build lookup dicts from CNN outputs
    pred_lookup  = dict(zip(all_names, all_preds))
    prob_lookup  = dict(zip(all_names, all_probs))
    label_lookup = dict(zip(all_names, all_labels))

    # Read GC and length from FASTA files
    seq_stats = {}  # name - {gc_content, sequence_length}

    for fasta_path, _ in test_fasta_pairs:
        current_name = None
        current_seq  = []

        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_name is not None:
                        seq = "".join(current_seq).upper().replace("T", "U")
                        n   = len(seq)
                        gc  = (seq.count("G") + seq.count("C")) / n \
                              if n > 0 else 0.0
                        seq_stats[current_name] = {
                            "gc_content":      gc,
                            "sequence_length": n,
                        }
                    current_name = line[1:].strip().split()[0]
                    current_seq  = []
                elif line:
                    current_seq.append(line)

        if current_name is not None:
            seq = "".join(current_seq).upper().replace("T", "U")
            n   = len(seq)
            gc  = (seq.count("G") + seq.count("C")) / n \
                  if n > 0 else 0.0
            seq_stats[current_name] = {
                "gc_content":      gc,
                "sequence_length": n,
            }

    # Build per-sequence rows
    rows = []

    for name in all_names:
        true  = label_lookup[name]
        pred  = pred_lookup[name]
        prob  = prob_lookup[name]
        etype = error_type(true, pred)
        stats = seq_stats.get(name, {"gc_content": None,
                                     "sequence_length": None})

        rows.append({
            "name":            name,
            "true_label":      "inherited" if true == 1 else "non_inherited",
            "predicted_label": "inherited" if pred == 1 else "non_inherited",
            "confidence":      prob,
            "error_type":      etype,
            "gc_content":      stats["gc_content"],
            "sequence_length": stats["sequence_length"],
        })

    # Save per-sequence file (A)
    seq_path = Path(output_dir) / "error_analysis.csv"
    header_a = ["name", "true_label", "predicted_label",
                 "confidence", "error_type",
                 "gc_content", "sequence_length"]

    with open(seq_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_a)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"  Per-sequence error file saved: {seq_path}")

    # Compute group-level summary (B)
    groups = ["TP", "TN", "FP", "FN"]

    summary_rows = []

    for group in groups:
        subset = [r for r in rows if r["error_type"] == group]
        n      = len(subset)

        if n == 0:
            summary_rows.append({
                "error_type": group,
                "count":      0,
                "gc_mean":    None,
                "gc_std":     None,
                "length_mean": None,
                "length_std":  None,
                "conf_mean":  None,
                "conf_std":   None,
            })
            continue

        gcs    = np.array([r["gc_content"]      for r in subset
                           if r["gc_content"] is not None])
        lens   = np.array([r["sequence_length"] for r in subset
                           if r["sequence_length"] is not None])
        confs  = np.array([r["confidence"]       for r in subset])

        summary_rows.append({
            "error_type":  group,
            "count":       n,
            "gc_mean":     float(np.mean(gcs))   if len(gcs)  > 0 else None,
            "gc_std":      float(np.std(gcs))    if len(gcs)  > 0 else None,
            "length_mean": float(np.mean(lens))  if len(lens) > 0 else None,
            "length_std":  float(np.std(lens))   if len(lens) > 0 else None,
            "conf_mean":   float(np.mean(confs)),
            "conf_std":    float(np.std(confs)),
        })

    # Confidence separation: correct vs incorrect
    correct   = np.array([r["confidence"] for r in rows
                          if r["error_type"] in ("TP", "TN")])
    incorrect = np.array([r["confidence"] for r in rows
                          if r["error_type"] in ("FP", "FN")])

    log.info("  Confidence separation:")
    if len(correct) > 0:
        log.info(f"    Correct   (TP+TN): mean={np.mean(correct):.4f}  "
                 f"std={np.std(correct):.4f}  n={len(correct)}")
    if len(incorrect) > 0:
        log.info(f"    Incorrect (FP+FN): mean={np.mean(incorrect):.4f}  "
                 f"std={np.std(incorrect):.4f}  n={len(incorrect)}")

    if len(correct) > 0 and len(incorrect) > 0:
        if np.mean(incorrect) > 0.7:
            log.warning(
                "  High-confidence errors detected — "
                "may indicate biological edge cases or bias"
            )
        else:
            log.info(
                "  Errors are low-confidence — "
                "model is uncertain on hard cases (expected)"
            )

    # Save summary file (B)
    summary_path = Path(output_dir) / "error_summary.csv"
    header_b     = ["error_type", "count",
                    "gc_mean", "gc_std",
                    "length_mean", "length_std",
                    "conf_mean", "conf_std"]

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_b)
        writer.writeheader()
        writer.writerows(summary_rows)

    log.info(f"  Group summary saved: {summary_path}")


def run_all_models(train_master, test_master,
                   inherited_train, non_inherited_train,
                   inherited_test, non_inherited_test,
                   output_dir="models_output",
                   epochs=50, batch_size=32,
                   learning_rate=0.001,
                   seed=42, skip_cnn=False):
    """
    Run all three models and save results.

    Arguments:
      train_master   — path to train_master.csv
      test_master    — path to test_master.csv
      inherited_train — path to inherited train FASTA
      non_inherited_train  — path to non inherited train FASTA
      inherited_test  — path to inherited test FASTA
      non_inherited_test   — path to non inherited test FASTA
      output_dir     — where to save all results
      epochs         — CNN training epochs
      batch_size     — CNN batch size
      learning_rate  — CNN learning rate
      seed           — random seed
      skip_cnn       — skip CNN training (quick LR+RF only)

    Returns:
      dictionary of all model metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Fix seeds
    set_seeds(seed)

    # Device
    device = torch.device("cpu")
    log.info(f"Device: {device}")

    log.info("Model Training")
    log.info("-" * 50)

    all_metrics = {}

    # Load feature matrices
    log.info("Loading feature matrices...")
    X_train, y_train, _, feature_names = load_feature_matrix(
        train_master
    )
    X_test, y_test, _, _ = load_feature_matrix(test_master)
    
    n_inherited_test = int(sum(y_test))
    n_non_inherited_test = len(y_test) - n_inherited_test
    log.info(f"Test set - inherited: {n_inherited_test}, "
             f"non-inherited: {n_non_inherited_test}")

    # Logistic Regression
    lr_metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test,
        feature_names,
        output_dir / "lr_results.txt"
    )
    all_metrics["Logistic Regression"] = lr_metrics

    # Random Forest
    rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test,
        feature_names,
        output_dir / "rf_results.txt"
    )
    all_metrics["Random Forest"] = rf_metrics

    # CNN
    if not skip_cnn:
        train_fasta_pairs = [
            (inherited_train, 1),
            (non_inherited_train,  0),
        ]
        test_fasta_pairs = [
            (inherited_test, 1),
            (non_inherited_test,  0),
        ]

        cnn_model, cnn_metrics, cnn_names, cnn_probs, cnn_labels = \
            train_cnn(
                train_fasta_pairs,
                test_fasta_pairs,
                output_dir / "cnn_results.txt",
                device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
        all_metrics["CNN"] = cnn_metrics

        # Saliency maps
        compute_saliency_maps(
            cnn_model,
            test_fasta_pairs,
            device,
            output_dir / "saliency_maps.csv"
        )

        # GC bias control
        compute_gc_vs_confidence(
            test_fasta_pairs,
            cnn_names,
            cnn_probs,
            cnn_labels,
            output_dir / "gc_vs_confidence.csv"
        )
        
        # Error analysis
        analyze_errors(
            test_fasta_pairs,
            cnn_names,
            [1 if p >= 0.5 else 0 for p in cnn_probs],
            cnn_probs,
            cnn_labels,
            test_master,
            output_dir,
        )

        # Shuffle control
        run_shuffle_control(
            train_fasta_pairs,
            test_fasta_pairs,
            device,
            output_dir / "shuffle_control.txt"
        )

    # Model comparison table
    comparison_path = output_dir / "model_comparison.csv"
    header = ["model", "f1", "auc", "precision", "recall"]

    with open(comparison_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for model_name, metrics in all_metrics.items():
            writer.writerow({
                "model":     model_name,
                "f1":        f"{metrics['f1']:.4f}",
                "auc":       f"{metrics['auc']:.4f}",
                "precision": f"{metrics['precision']:.4f}",
                "recall":    f"{metrics['recall']:.4f}",
            })

    log.info(f"Model comparison saved to {comparison_path}")
    log.info("All models complete")

    return all_metrics


# run

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Train and evaluate all models"
    )
    parser.add_argument(
        "--train_master",
        type=str,
        required=True,
        help="Path to train_master.csv"
    )
    parser.add_argument(
        "--test_master",
        type=str,
        required=True,
        help="Path to test_master.csv"
    )
    parser.add_argument(
        "--inherited_train",
        type=str,
        required=True,
        help="Path to inherited train FASTA"
    )
    parser.add_argument(
        "--non_inherited_train",
        type=str,
        required=True,
        help="Path to non_inherited train FASTA"
    )
    parser.add_argument(
        "--inherited_test",
        type=str,
        required=True,
        help="Path to inherited test FASTA"
    )
    parser.add_argument(
        "--non_inherited_test",
        type=str,
        required=True,
        help="Path to non_inherited test FASTA"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="models_output",
        help="Output directory for all model results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="CNN training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="CNN batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="CNN learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--skip_cnn",
        action="store_true",
        help="Skip CNN training (quick LR+RF only run)"
    )
    args = parser.parse_args()

    run_all_models(
        train_master=args.train_master,
        test_master=args.test_master,
        inherited_train=args.inherited_train,
        non_inherited_train=args.non_inherited_train,
        inherited_test=args.inherited_test,
        non_inherited_test=args.non_inherited_test,
        output_dir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        skip_cnn=args.skip_cnn
    )

    log.info(f"Total runtime: {time.time() - start:.2f} seconds")

