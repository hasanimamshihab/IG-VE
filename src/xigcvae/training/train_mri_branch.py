#!/usr/bin/env python3
"""
Train final MRI branch using paired GE+MRI and MRI-only structural MRI scans.

Input:
  data/Paired_data_metadata.csv
  data/mri_only_metadata.csv
  data/paired_MRI/*.nii.gz
  data/only_MRI/*.nii.gz

Model:
  3D structural MRI -> MFV128 -> AD/CN classifier

Outputs:
  outputs/mri_branch/final_mri_branch_model.pt
  outputs/mri_branch/final_mri_branch_config.json
  outputs/mri_branch/final_mri_branch_metrics.csv
  outputs/mri_branch/final_mri_branch_predictions.csv
  outputs/mri_branch/final_mri_branch_training_history.csv
  outputs/mri_branch/mfv_features_all_1495_mri_subjects.csv
  outputs/mri_branch/mfv_features_paired_289_subjects.csv
  outputs/mri_branch/mfv_features_mri_only_1206_subjects.csv
  outputs/tables/supplementary/Table_S7_MRI_branch_metrics.xlsx
  manifests/mri_branch_final_split_seed_42.csv
"""

from __future__ import annotations

import copy
import json
import random
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import zoom

from src.xigcvae.models.mri_branch import MRI3DBranch
from src.xigcvae.utils.excel_style import style_supplementary_table


warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Configuration
# -----------------------------

SEED = 42
MFV_DIM = 128

BATCH_SIZE = 4
MAX_EPOCHS = 300
PATIENCE = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.30
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 0
TARGET_SHAPE = (64, 64, 64)

PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs" / "mri_branch"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "supplementary"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

PAIRED_META_FILE = DATA_DIR / "Paired_data_metadata.csv"
MRI_ONLY_META_FILE = DATA_DIR / "mri_only_metadata.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utility functions
# -----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def resolve_path(path_value):
    if pd.isna(path_value):
        return None

    p = Path(str(path_value))

    if p.is_absolute():
        return p

    return PROJECT_ROOT / p


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def compute_metrics(y_true, y_prob, y_pred):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "Subjects, n": int(len(y_true)),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "AUC": safe_auc(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall / AD sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "CN specificity": specificity,
        "F1 score": f1_score(y_true, y_pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def robust_normalize_mri(data: np.ndarray) -> np.ndarray:
    """
    Per-image robust normalization.

    Uses nonzero finite voxels as brain/intensity mask.
    Clips nonzero voxels to 1st and 99th percentiles.
    Z-scores using clipped nonzero voxels.
    Keeps background as 0.
    """
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    mask = np.isfinite(data) & (data != 0)

    if mask.sum() < 100:
        return np.zeros_like(data, dtype=np.float32)

    vals = data[mask]
    lo, hi = np.percentile(vals, [1, 99])

    clipped = np.clip(data, lo, hi)
    vals_clip = clipped[mask]

    mean = vals_clip.mean()
    std = vals_clip.std()

    if std < 1e-6:
        std = 1.0

    normed = (clipped - mean) / std
    normed[~mask] = 0.0

    return normed.astype(np.float32)


def resize_mri(data: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    """
    Resize MRI volume to fixed 64 x 64 x 64 for safer 3D CNN training.
    """
    if tuple(data.shape) == tuple(target_shape):
        return data.astype(np.float32)

    factors = [t / s for t, s in zip(target_shape, data.shape)]
    resized = zoom(data, zoom=factors, order=1)
    return resized.astype(np.float32)



# -----------------------------
# Data
# -----------------------------

def build_mri_dataframe() -> pd.DataFrame:
    paired = pd.read_csv(PAIRED_META_FILE)
    mri_only = pd.read_csv(MRI_ONLY_META_FILE)

    paired["has_mri"] = bool_series(paired["has_mri"])
    paired["ge_column_found"] = bool_series(paired["ge_column_found"])
    paired["diagnosis_binary"] = paired["diagnosis_binary"].astype(int)

    mri_only["has_mri"] = bool_series(mri_only["has_mri"])
    mri_only["diagnosis_binary"] = mri_only["diagnosis_binary"].astype(int)

    paired_mri = paired[
        (paired["diagnosis"].isin(["AD", "CN"])) &
        (paired["ge_column_found"]) &
        (paired["has_mri"])
    ].copy()

    mri_only = mri_only[
        (mri_only["diagnosis"].isin(["AD", "CN"])) &
        (mri_only["has_mri"])
    ].copy()

    paired_mri["mri_cohort"] = "Paired_GE_MRI"
    mri_only["mri_cohort"] = "MRI_only"

    common_cols = [
        "subject_id",
        "bids_id",
        "rid",
        "diagnosis",
        "diagnosis_binary",
        "age",
        "sex",
        "has_measured_ge",
        "has_mri",
        "mri_rel_path",
        "mri_abs_path",
        "source_group",
        "mri_cohort",
    ]

    for col in common_cols:
        if col not in paired_mri.columns:
            paired_mri[col] = np.nan
        if col not in mri_only.columns:
            mri_only[col] = np.nan

    df = pd.concat(
        [
            paired_mri[common_cols],
            mri_only[common_cols],
        ],
        ignore_index=True,
        sort=False,
    )

    df["mri_path_resolved"] = df["mri_abs_path"].apply(resolve_path).astype(str)
    df["mri_file_exists"] = df["mri_path_resolved"].apply(lambda x: Path(x).exists())

    missing = df[~df["mri_file_exists"]]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"{len(missing)} MRI files are missing. Examples:\n"
            + "\n".join(missing["mri_path_resolved"].head(10).tolist())
        )

    df["mri_subject_key"] = (
        df["mri_cohort"].astype(str)
        + "__"
        + df["subject_id"].astype(str)
        + "__"
        + df["bids_id"].astype(str)
    )

    return df


class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, return_label: bool = True):
        self.df = df.reset_index(drop=True).copy()
        self.return_label = return_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["mri_path_resolved"]

        img = nib.load(path)
        data = np.asanyarray(img.dataobj).astype(np.float32)
        data = robust_normalize_mri(data)
        data = resize_mri(data)

        x = torch.from_numpy(data[None, ...].astype(np.float32))

        if self.return_label:
            y = int(row["diagnosis_binary"])
            return x, torch.tensor(y, dtype=torch.long), idx

        return x, idx


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    idx = np.arange(len(df))

    # Stratify jointly by MRI cohort and diagnosis so paired/MRI-only and AD/CN
    # distributions are preserved across train/validation/test.
    stratify_key = df["mri_cohort"].astype(str) + "_" + df["diagnosis"].astype(str)

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=SEED,
        stratify=stratify_key,
    )

    temp_stratify = stratify_key.iloc[temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_stratify,
    )

    split = np.array([""] * len(df), dtype=object)
    split[train_idx] = "Train"
    split[val_idx] = "Validation"
    split[test_idx] = "Test"

    out = df.copy()
    out["split"] = split

    return out


# -----------------------------
# Training and prediction
# -----------------------------

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_y = []
    all_prob = []
    all_pred = []

    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits, _ = model(xb)
            loss = criterion(logits, yb)

            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            total_loss += loss.item() * len(yb)
            all_y.append(yb.cpu().numpy())
            all_prob.append(prob.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y = np.concatenate(all_y)
    prob = np.concatenate(all_prob)
    pred = np.concatenate(all_pred)

    metrics = compute_metrics(y, prob, pred)
    metrics["Loss"] = total_loss / len(y)

    return metrics


def predict_features(model, df: pd.DataFrame, device):
    dataset = MRIDataset(df, return_label=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model.eval()

    all_prob = np.zeros(len(df), dtype=np.float32)
    all_pred = np.zeros(len(df), dtype=np.int64)
    all_mfv = np.zeros((len(df), MFV_DIM), dtype=np.float32)

    with torch.no_grad():
        for xb, yb, indices in loader:
            xb = xb.to(device, non_blocking=True)

            logits, mfv = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)

            idx_np = indices.numpy()
            all_prob[idx_np] = prob.cpu().numpy()
            all_pred[idx_np] = pred.cpu().numpy()
            all_mfv[idx_np, :] = mfv.cpu().numpy()

    return all_prob, all_pred, all_mfv


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safer for HPC/CUDA low-level 3D convolution stability.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("========== TRAIN MRI BRANCH ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("======================================\n")

    df = build_mri_dataframe()
    df = make_split(df)

    split_manifest_path = MANIFEST_DIR / "mri_branch_final_split_seed_42.csv"
    df.to_csv(split_manifest_path, index=False)

    print("========== MRI BRANCH INPUT ==========")
    print(f"Total MRI subjects: {len(df):,}")
    print(df.groupby(["mri_cohort", "diagnosis"]).size().unstack(fill_value=0))
    print()
    print("Split counts:")
    print(df.groupby(["split", "diagnosis"]).size().unstack(fill_value=0))
    print()
    print("Split counts by MRI cohort:")
    print(df.groupby(["split", "mri_cohort", "diagnosis"]).size().unstack(fill_value=0))
    print("======================================\n")

    train_df = df[df["split"] == "Train"].reset_index(drop=True)
    val_df = df[df["split"] == "Validation"].reset_index(drop=True)
    test_df = df[df["split"] == "Test"].reset_index(drop=True)

    train_dataset = MRIDataset(train_df, return_label=True)
    val_dataset = MRIDataset(val_df, return_label=True)
    test_dataset = MRIDataset(test_df, return_label=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )

    model = MRI3DBranch(
        mfv_dim=MFV_DIM,
        dropout=DROPOUT,
        num_classes=2,
    ).to(device)

    y_train = train_df["diagnosis_binary"].astype(int).values
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2.0 * np.maximum(class_counts, 1))
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_epoch = 0
    best_score = (-np.inf, -np.inf, np.inf)
    patience_counter = 0
    history = []


    # Resume support for multi-session MRI training
    start_epoch = 1
    last_checkpoint_path = OUT_DIR / "checkpoint_last_mri_branch_training.pt"

    if last_checkpoint_path.exists():
        print(f"Resuming MRI branch training from checkpoint: {last_checkpoint_path}")
        ckpt = torch.load(last_checkpoint_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        best_state = ckpt.get("best_state_dict", ckpt["model_state_dict"])
        best_epoch = int(ckpt.get("best_epoch", 0))
        best_score = tuple(ckpt.get("best_score", best_score))
        patience_counter = int(ckpt.get("patience_counter", 0))
        history = ckpt.get("history", [])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

        print(f"Resume start epoch: {start_epoch}")
        print(f"Best epoch so far: {best_epoch}")
        print(f"Patience counter so far: {patience_counter}")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(yb)
            train_n += len(yb)

        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["Loss"],
            "train_accuracy": train_metrics["Accuracy"],
            "train_balanced_accuracy": train_metrics["Balanced accuracy"],
            "train_auc": train_metrics["AUC"],
            "train_f1": train_metrics["F1 score"],
            "validation_loss": val_metrics["Loss"],
            "validation_accuracy": val_metrics["Accuracy"],
            "validation_balanced_accuracy": val_metrics["Balanced accuracy"],
            "validation_auc": val_metrics["AUC"],
            "validation_f1": val_metrics["F1 score"],
        }
        history.append(row)

        current_score = (
            val_metrics["AUC"] if not np.isnan(val_metrics["AUC"]) else -np.inf,
            val_metrics["Balanced accuracy"],
            -val_metrics["Loss"],
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        torch.save(
            {
                "checkpoint_type": "last_epoch",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_score": list(best_score),
                "patience_counter": patience_counter,
                "history": history,
                "mfv_dim": MFV_DIM,
                "dropout": DROPOUT,
                "seed": SEED,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
            },
            last_checkpoint_path,
        )

        if True:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_metrics['Loss']:.4f} | "
                f"val loss {val_metrics['Loss']:.4f} | "
                f"val AUC {val_metrics['AUC']:.3f} | "
                f"val balanced acc {val_metrics['Balanced accuracy']:.3f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was recorded.")

    model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    metrics_rows = []

    for split_name, split_df, metrics in [
        ("Train", train_df, train_metrics),
        ("Validation", val_df, val_metrics),
        ("Test", test_df, test_metrics),
    ]:
        row = {
            "Model": "Final MRI branch",
            "Input representation": "3D structural MRI -> MFV128",
            "MFV dimension": MFV_DIM,
            "Split": split_name,
            "Subjects, n": len(split_df),
            "Best epoch": best_epoch,
            "Loss": metrics["Loss"],
            "Accuracy": metrics["Accuracy"],
            "Balanced accuracy": metrics["Balanced accuracy"],
            "AUC": metrics["AUC"],
            "Precision": metrics["Precision"],
            "Recall / AD sensitivity": metrics["Recall / AD sensitivity"],
            "CN specificity": metrics["CN specificity"],
            "F1 score": metrics["F1 score"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"],
        }
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    all_prob, all_pred, all_mfv = predict_features(model, df.reset_index(drop=True), device)

    pred_df = df.reset_index(drop=True).copy()
    pred_df["true_label"] = pred_df["diagnosis_binary"].astype(int)
    pred_df["predicted_AD_probability"] = all_prob
    pred_df["predicted_label"] = all_pred
    pred_df["predicted_diagnosis"] = np.where(all_pred == 1, "AD", "CN")

    mfv_cols = [f"MFV{j+1:03d}" for j in range(MFV_DIM)]
    mfv_df = pd.concat(
        [
            pred_df[
                [
                    "mri_subject_key",
                    "subject_id",
                    "bids_id",
                    "rid",
                    "diagnosis",
                    "diagnosis_binary",
                    "mri_cohort",
                    "split",
                    "mri_rel_path",
                    "mri_abs_path",
                ]
            ].copy(),
            pd.DataFrame(all_mfv, columns=mfv_cols),
        ],
        axis=1,
    )

    mfv_paired = mfv_df[mfv_df["mri_cohort"] == "Paired_GE_MRI"].copy()
    mfv_mri_only = mfv_df[mfv_df["mri_cohort"] == "MRI_only"].copy()

    model_path = OUT_DIR / "final_mri_branch_model.pt"
    config_path = OUT_DIR / "final_mri_branch_config.json"
    metrics_path = OUT_DIR / "final_mri_branch_metrics.csv"
    predictions_path = OUT_DIR / "final_mri_branch_predictions.csv"
    history_path = OUT_DIR / "final_mri_branch_training_history.csv"
    mfv_all_path = OUT_DIR / "mfv_features_all_1495_mri_subjects.csv"
    mfv_paired_path = OUT_DIR / "mfv_features_paired_289_subjects.csv"
    mfv_mri_only_path = OUT_DIR / "mfv_features_mri_only_1206_subjects.csv"
    table_s7_path = TABLE_DIR / "Table_S7_MRI_branch_metrics.xlsx"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mfv_dim": MFV_DIM,
            "dropout": DROPOUT,
            "best_epoch": best_epoch,
            "seed": SEED,
            "model_class": "MRI3DBranch",
            "normalization": "Per-image nonzero brain robust clipping p1-p99 followed by z-score; background set to zero; resized to 64x64x64",
        },
        model_path,
    )

    config = {
        "model": "Final MRI branch",
        "input": "3D structural MRI",
        "output": "MFV128",
        "seed": SEED,
        "mri_subjects_total": int(len(df)),
        "paired_ge_mri_subjects": int((df["mri_cohort"] == "Paired_GE_MRI").sum()),
        "mri_only_subjects": int((df["mri_cohort"] == "MRI_only").sum()),
        "mfv_dimension": MFV_DIM,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "label_smoothing": LABEL_SMOOTHING,
        "split": "70/15/15 stratified by MRI cohort and diagnosis",
        "best_epoch": int(best_epoch),
        "normalization": "Per-image robust normalization using nonzero voxels; clip p1-p99; z-score; background=0; resized to 64x64x64",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(predictions_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    mfv_df.to_csv(mfv_all_path, index=False)
    mfv_paired.to_csv(mfv_paired_path, index=False)
    mfv_mri_only.to_csv(mfv_mri_only_path, index=False)

    with pd.ExcelWriter(table_s7_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, index=False, sheet_name="MRI branch metrics")

    style_supplementary_table(table_s7_path, "MRI branch metrics", freeze_panes="A2", max_width=36)

    print("\n========== FINAL MRI BRANCH METRICS ==========")
    print(f"Total MRI subjects: {len(df)}")
    print(f"Paired GE+MRI subjects: {(df['mri_cohort'] == 'Paired_GE_MRI').sum()}")
    print(f"MRI-only subjects: {(df['mri_cohort'] == 'MRI_only').sum()}")
    print(f"MFV dimension: {MFV_DIM}")
    print(f"Best epoch: {best_epoch}")
    print()
    print(metrics_df.to_string(index=False))
    print("=============================================\n")

    print("Saved:")
    for p in [
        model_path,
        config_path,
        metrics_path,
        predictions_path,
        history_path,
        mfv_all_path,
        mfv_paired_path,
        mfv_mri_only_path,
        table_s7_path,
        split_manifest_path,
    ]:
        print(p)


if __name__ == "__main__":
    main()
