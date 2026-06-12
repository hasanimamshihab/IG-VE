from __future__ import annotations

import copy
import json
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.xigcvae.models.aligned_gfv_model import AlignedGFVModel


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

PAIRED_TABLE = PROJECT_ROOT / "outputs/xigcvae/paired_gfv128_mfv128_289_subjects.csv"
REAL_GFV_302_FILE = PROJECT_ROOT / "outputs/gene_branch/real_gfv_features_302_subjects.csv"
MRI_ONLY_MFV_FILE = PROJECT_ROOT / "outputs/mri_branch/mfv_features_mri_only_1206_subjects.csv"

OUT_DIR = PROJECT_ROOT / "outputs/aligned_gfv"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
GFV_DIM = 128
MFV_DIM = 128
AGFV_DIM = 128

BATCH_SIZE = 24
MAX_EPOCHS = 500
PATIENCE = 80
LR = 3e-4
WEIGHT_DECAY = 5e-4
CONTRASTIVE_TEMP = 0.20

CONFIGS = [
    {
        "name": "agfv_h256_align050_contrast010",
        "hidden_dim": 256,
        "dropout": 0.20,
        "cls_g_weight": 1.00,
        "cls_m_weight": 1.00,
        "align_weight": 0.50,
        "contrast_weight": 0.10,
        "corr_weight": 0.10,
        "ge_recon_weight": 0.10,
        "mri_to_gfv_weight": 0.02,
    },
    {
        "name": "agfv_h256_align100_contrast010",
        "hidden_dim": 256,
        "dropout": 0.20,
        "cls_g_weight": 1.00,
        "cls_m_weight": 1.00,
        "align_weight": 1.00,
        "contrast_weight": 0.10,
        "corr_weight": 0.10,
        "ge_recon_weight": 0.10,
        "mri_to_gfv_weight": 0.02,
    },
    {
        "name": "agfv_h128_align100_contrast020",
        "hidden_dim": 128,
        "dropout": 0.20,
        "cls_g_weight": 1.00,
        "cls_m_weight": 1.00,
        "align_weight": 1.00,
        "contrast_weight": 0.20,
        "corr_weight": 0.10,
        "ge_recon_weight": 0.10,
        "mri_to_gfv_weight": 0.02,
    },
    {
        "name": "agfv_h256_align100_no_mritogfv",
        "hidden_dim": 256,
        "dropout": 0.20,
        "cls_g_weight": 1.00,
        "cls_m_weight": 1.00,
        "align_weight": 1.00,
        "contrast_weight": 0.10,
        "corr_weight": 0.10,
        "ge_recon_weight": 0.10,
        "mri_to_gfv_weight": 0.00,
    },
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_feature_cols(df: pd.DataFrame, prefix: str):
    pat = re.compile(rf"^{prefix}\d+$", re.IGNORECASE)
    cols = [c for c in df.columns if pat.match(c)]
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0]))


def safe_meta_cols(df: pd.DataFrame):
    preferred = [
        "subject_id",
        "bids_id",
        "rid",
        "diagnosis",
        "diagnosis_binary",
        "split",
        "mri_cohort",
        "source_group",
        "has_measured_ge",
        "has_mri",
        "mri_rel_path",
        "mri_abs_path",
    ]
    return [c for c in preferred if c in df.columns]


class PairedDataset(Dataset):
    def __init__(self, gfv, mfv, y):
        self.gfv = torch.tensor(gfv, dtype=torch.float32)
        self.mfv = torch.tensor(mfv, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.gfv[idx], self.mfv[idx], self.y[idx]


def contrastive_loss(z_m, z_g, temperature=0.20):
    z_m = F.normalize(z_m, dim=1)
    z_g = F.normalize(z_g, dim=1)

    logits = torch.matmul(z_m, z_g.T) / temperature
    labels = torch.arange(z_m.shape[0], device=z_m.device)

    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def correlation_loss(z_m, z_g, eps=1e-8):
    z_m = z_m - z_m.mean(dim=1, keepdim=True)
    z_g = z_g - z_g.mean(dim=1, keepdim=True)

    numerator = (z_m * z_g).sum(dim=1)
    denominator = torch.sqrt((z_m.pow(2).sum(dim=1) + eps) * (z_g.pow(2).sum(dim=1) + eps))

    corr = numerator / denominator
    return 1.0 - corr.mean()


def matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    subject_corrs = []
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        if np.std(a) > 0 and np.std(b) > 0:
            subject_corrs.append(np.corrcoef(a, b)[0, 1])

    feature_corrs = []
    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]
        if np.std(a) > 0 and np.std(b) > 0:
            feature_corrs.append(np.corrcoef(a, b)[0, 1])

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "mean_subject_r": float(np.mean(subject_corrs)) if subject_corrs else np.nan,
        "median_subject_r": float(np.median(subject_corrs)) if subject_corrs else np.nan,
        "mean_feature_r": float(np.mean(feature_corrs)) if feature_corrs else np.nan,
        "median_feature_r": float(np.median(feature_corrs)) if feature_corrs else np.nan,
    }


def cls_metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = np.nan

    acc = accuracy_score(y_true, pred)
    bacc = balanced_accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "precision": float(prec),
        "recall_ad_sensitivity": float(rec),
        "cn_specificity": float(spec),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def encode_batches(model, x_scaled, encoder_name, device):
    model.eval()
    outs = []
    probs = []

    for start in range(0, x_scaled.shape[0], 256):
        end = min(start + 256, x_scaled.shape[0])
        x = torch.tensor(x_scaled[start:end], dtype=torch.float32, device=device)

        if encoder_name == "gfv":
            z = model.encode_gfv(x)
        elif encoder_name == "mfv":
            z = model.encode_mfv(x)
        else:
            raise ValueError(encoder_name)

        logits = model.classifier(z)
        prob = torch.softmax(logits, dim=1)[:, 1]

        outs.append(z.cpu().numpy())
        probs.append(prob.cpu().numpy())

    return np.vstack(outs), np.concatenate(probs)


@torch.no_grad()
def evaluate_split(model, gfv_scaled, mfv_scaled, y, device):
    z_g, prob_g = encode_batches(model, gfv_scaled, "gfv", device)
    z_m, prob_m = encode_batches(model, mfv_scaled, "mfv", device)

    align = matrix_metrics(z_g, z_m)

    cosine = np.sum(z_g * z_m, axis=1) / (
        np.linalg.norm(z_g, axis=1) * np.linalg.norm(z_m, axis=1) + 1e-8
    )

    ge_cls = cls_metrics(y, prob_g)
    mri_cls = cls_metrics(y, prob_m)

    return {
        "alignment_mse": align["mse"],
        "alignment_mae": align["mae"],
        "alignment_r2": align["r2"],
        "alignment_subject_r": align["mean_subject_r"],
        "alignment_feature_r": align["mean_feature_r"],
        "alignment_cosine_mean": float(np.mean(cosine)),
        "alignment_cosine_median": float(np.median(cosine)),
        "ge_auc": ge_cls["auc"],
        "ge_balanced_accuracy": ge_cls["balanced_accuracy"],
        "mri_auc": mri_cls["auc"],
        "mri_accuracy": mri_cls["accuracy"],
        "mri_balanced_accuracy": mri_cls["balanced_accuracy"],
        "mri_precision": mri_cls["precision"],
        "mri_recall_ad_sensitivity": mri_cls["recall_ad_sensitivity"],
        "mri_cn_specificity": mri_cls["cn_specificity"],
        "mri_f1": mri_cls["f1"],
        "mri_tn": mri_cls["tn"],
        "mri_fp": mri_cls["fp"],
        "mri_fn": mri_cls["fn"],
        "mri_tp": mri_cls["tp"],
        "z_g": z_g,
        "z_m": z_m,
        "prob_g": prob_g,
        "prob_m": prob_m,
    }


def compute_loss(outputs, gfv, y, cfg, class_weight=None):
    z_g = outputs["z_g"]
    z_m = outputs["z_m"]

    cls_g = F.cross_entropy(outputs["logits_g"], y, weight=class_weight)
    cls_m = F.cross_entropy(outputs["logits_m"], y, weight=class_weight)

    align = F.mse_loss(F.normalize(z_m, dim=1), F.normalize(z_g.detach(), dim=1))
    corr = correlation_loss(z_m, z_g.detach())
    contr = contrastive_loss(z_m, z_g.detach(), temperature=CONTRASTIVE_TEMP)

    ge_recon = F.mse_loss(outputs["recon_g"], gfv)
    mri_to_gfv = F.mse_loss(outputs["pred_g_from_m"], gfv)

    total = (
        cfg["cls_g_weight"] * cls_g
        + cfg["cls_m_weight"] * cls_m
        + cfg["align_weight"] * align
        + cfg["contrast_weight"] * contr
        + cfg["corr_weight"] * corr
        + cfg["ge_recon_weight"] * ge_recon
        + cfg["mri_to_gfv_weight"] * mri_to_gfv
    )

    return total, {
        "cls_g": cls_g,
        "cls_m": cls_m,
        "align": align,
        "corr": corr,
        "contrast": contr,
        "ge_recon": ge_recon,
        "mri_to_gfv": mri_to_gfv,
    }


def train_one_config(cfg, data, device):
    set_seed(SEED)

    model = AlignedGFVModel(
        gfv_dim=GFV_DIM,
        mfv_dim=MFV_DIM,
        agfv_dim=AGFV_DIM,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", LR), weight_decay=cfg.get("weight_decay", WEIGHT_DECAY))

    train_loader = data["train_loader"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    gfv_scaled = data["gfv_scaled"]
    mfv_scaled = data["mfv_scaled"]
    y = data["y"]

    class_weight = None
    if cfg.get("balanced_loss", False):
        y_train = y[train_idx]
        counts = np.bincount(y_train, minlength=2).astype(float)
        weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
        class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    best_state = None
    best_epoch = 0
    best_score = -np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        losses = []
        cls_m_losses = []
        align_losses = []
        contrast_losses = []

        for gfv, mfv, yy in train_loader:
            gfv = gfv.to(device)
            mfv = mfv.to(device)
            yy = yy.to(device)

            optimizer.zero_grad()

            outputs = model(gfv, mfv)
            total_loss, parts = compute_loss(outputs, gfv, yy, cfg, class_weight=class_weight)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(total_loss.item())
            cls_m_losses.append(parts["cls_m"].item())
            align_losses.append(parts["align"].item())
            contrast_losses.append(parts["contrast"].item())

        train_metrics = evaluate_split(
            model,
            gfv_scaled[train_idx],
            mfv_scaled[train_idx],
            y[train_idx],
            device,
        )

        val_metrics = evaluate_split(
            model,
            gfv_scaled[val_idx],
            mfv_scaled[val_idx],
            y[val_idx],
            device,
        )

        # Primary: MRI-derived aligned feature disease performance.
        # Secondary: paired MRI-GE alignment.
        # Penalize train-validation gap to reduce overfitting.
        generalization_gap = max(0.0, train_metrics["mri_auc"] - val_metrics["mri_auc"]) + max(
            0.0,
            train_metrics["mri_balanced_accuracy"] - val_metrics["mri_balanced_accuracy"],
        )

        score = (
            val_metrics["mri_auc"]
            + val_metrics["mri_balanced_accuracy"]
            + 0.50 * val_metrics["alignment_cosine_mean"]
            + 0.10 * val_metrics["alignment_subject_r"]
            - 0.15 * generalization_gap
        )

        row = {
            "config": cfg["name"],
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_mri_auc": train_metrics["mri_auc"],
            "train_mri_balanced_accuracy": train_metrics["mri_balanced_accuracy"],
            "train_alignment_cosine_mean": train_metrics["alignment_cosine_mean"],
            "val_mri_auc": val_metrics["mri_auc"],
            "val_mri_balanced_accuracy": val_metrics["mri_balanced_accuracy"],
            "val_alignment_cosine_mean": val_metrics["alignment_cosine_mean"],
            "val_alignment_subject_r": val_metrics["alignment_subject_r"],
            "selection_score": score,
        }
        history.append(row)

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "config": copy.deepcopy(cfg),
                "epoch": int(epoch),
                "selection_metric": "val_mri_auc + val_mri_balanced_accuracy + 0.50*val_alignment_cosine + 0.10*val_alignment_subject_r - 0.15*generalization_gap",
                "selection_score": float(score),
                "val_mri_auc": float(val_metrics["mri_auc"]),
                "val_mri_balanced_accuracy": float(val_metrics["mri_balanced_accuracy"]),
                "val_alignment_cosine_mean": float(val_metrics["alignment_cosine_mean"]),
                "val_alignment_subject_r": float(val_metrics["alignment_subject_r"]),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{cfg['name']} | Epoch {epoch:03d} | "
                f"train MRI AUC {train_metrics['mri_auc']:.3f} | "
                f"val MRI AUC {val_metrics['mri_auc']:.3f} | "
                f"val MRI bacc {val_metrics['mri_balanced_accuracy']:.3f} | "
                f"val align cosine {val_metrics['alignment_cosine_mean']:.3f} | "
                f"score {score:.3f}"
            )

        if patience_counter >= PATIENCE:
            print(f"{cfg['name']} early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No best state saved for {cfg['name']}")

    model.load_state_dict(best_state["model_state_dict"])

    return {
        "config": cfg,
        "model": model,
        "best_state": best_state,
        "history": pd.DataFrame(history),
    }


def metric_row(cfg, split_name, metrics, n, best_epoch):
    return {
        "Model": "MRI-aligned molecular representation",
        "Config": cfg["name"],
        "Split": split_name,
        "Subjects, n": int(n),
        "Best epoch": int(best_epoch),
        "MRI-derived AGFV AUC": metrics["mri_auc"],
        "MRI-derived AGFV balanced accuracy": metrics["mri_balanced_accuracy"],
        "MRI-derived AGFV accuracy": metrics["mri_accuracy"],
        "MRI-derived AGFV AD sensitivity": metrics["mri_recall_ad_sensitivity"],
        "MRI-derived AGFV CN specificity": metrics["mri_cn_specificity"],
        "MRI-derived AGFV F1": metrics["mri_f1"],
        "GE-derived AGFV AUC": metrics["ge_auc"],
        "GE-derived AGFV balanced accuracy": metrics["ge_balanced_accuracy"],
        "MRI-GE alignment cosine mean": metrics["alignment_cosine_mean"],
        "MRI-GE alignment cosine median": metrics["alignment_cosine_median"],
        "MRI-GE alignment subject-wise r": metrics["alignment_subject_r"],
        "MRI-GE alignment feature-wise r": metrics["alignment_feature_r"],
        "MRI-GE alignment MSE": metrics["alignment_mse"],
        "TN": metrics["mri_tn"],
        "FP": metrics["mri_fp"],
        "FN": metrics["mri_fn"],
        "TP": metrics["mri_tp"],
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== TRAIN MRI-ALIGNED GFV MODEL ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Paired table: {PAIRED_TABLE}")
    print(f"Real GFV 302 file: {REAL_GFV_302_FILE}")
    print(f"MRI-only MFV file: {MRI_ONLY_MFV_FILE}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=================================================\n")

    for f in [PAIRED_TABLE, REAL_GFV_302_FILE, MRI_ONLY_MFV_FILE]:
        if not f.exists():
            raise FileNotFoundError(f)

    paired = pd.read_csv(PAIRED_TABLE)
    real_gfv_302 = pd.read_csv(REAL_GFV_302_FILE)
    mri_only = pd.read_csv(MRI_ONLY_MFV_FILE)

    gfv_cols = get_feature_cols(paired, "GFV")
    mfv_cols = get_feature_cols(paired, "MFV")

    if len(gfv_cols) != GFV_DIM:
        raise ValueError(f"Expected 128 GFV columns, found {len(gfv_cols)}")
    if len(mfv_cols) != MFV_DIM:
        raise ValueError(f"Expected 128 MFV columns, found {len(mfv_cols)}")

    for c in gfv_cols:
        if c not in real_gfv_302.columns:
            raise ValueError(f"Missing GFV column in real GFV 302 file: {c}")
    for c in mfv_cols:
        if c not in mri_only.columns:
            raise ValueError(f"Missing MFV column in MRI-only file: {c}")

    print("Paired table shape:", paired.shape)
    print("Real GFV 302 shape:", real_gfv_302.shape)
    print("MRI-only MFV shape:", mri_only.shape)
    print("\nPaired split counts:")
    print(paired["split"].value_counts())
    print("\nPaired split x diagnosis:")
    print(pd.crosstab(paired["split"], paired["diagnosis"]))
    print()

    train_idx = np.where(paired["split"].values == "Train")[0]
    val_idx = np.where(paired["split"].values == "Validation")[0]
    test_idx = np.where(paired["split"].values == "Test")[0]

    y = paired["diagnosis_binary"].astype(int).values

    gfv_raw = paired[gfv_cols].astype(float).values
    mfv_raw = paired[mfv_cols].astype(float).values

    gfv_scaler = StandardScaler()
    mfv_scaler = StandardScaler()

    gfv_scaled = np.zeros_like(gfv_raw, dtype=np.float32)
    mfv_scaled = np.zeros_like(mfv_raw, dtype=np.float32)

    gfv_scaled[train_idx] = gfv_scaler.fit_transform(gfv_raw[train_idx])
    gfv_scaled[val_idx] = gfv_scaler.transform(gfv_raw[val_idx])
    gfv_scaled[test_idx] = gfv_scaler.transform(gfv_raw[test_idx])

    mfv_scaled[train_idx] = mfv_scaler.fit_transform(mfv_raw[train_idx])
    mfv_scaled[val_idx] = mfv_scaler.transform(mfv_raw[val_idx])
    mfv_scaled[test_idx] = mfv_scaler.transform(mfv_raw[test_idx])

    train_ds = PairedDataset(gfv_scaled[train_idx], mfv_scaled[train_idx], y[train_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    data = {
        "train_loader": train_loader,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "gfv_scaled": gfv_scaled,
        "mfv_scaled": mfv_scaled,
        "y": y,
    }

    all_metrics = []
    all_histories = []
    best_result = None
    best_score = -np.inf

    for cfg in CONFIGS:
        print("\n==================================================")
        print("Training config:", cfg)
        print("==================================================")

        result = train_one_config(cfg, data, device)
        cfg = result["config"]
        state = result["best_state"]
        model = result["model"]

        train_m = evaluate_split(model, gfv_scaled[train_idx], mfv_scaled[train_idx], y[train_idx], device)
        val_m = evaluate_split(model, gfv_scaled[val_idx], mfv_scaled[val_idx], y[val_idx], device)
        test_m = evaluate_split(model, gfv_scaled[test_idx], mfv_scaled[test_idx], y[test_idx], device)

        all_metrics.extend(
            [
                metric_row(cfg, "Train", train_m, len(train_idx), state["epoch"]),
                metric_row(cfg, "Validation", val_m, len(val_idx), state["epoch"]),
                metric_row(cfg, "Test", test_m, len(test_idx), state["epoch"]),
            ]
        )

        all_histories.append(result["history"])

        score = state["selection_score"]
        if score > best_score:
            best_score = score
            best_result = {
                "config": cfg,
                "state": state,
                "model": model,
                "train_metrics": train_m,
                "val_metrics": val_m,
                "test_metrics": test_m,
            }

    if best_result is None:
        raise RuntimeError("No best aligned GFV model selected")

    best_cfg = best_result["config"]
    best_state = best_result["state"]
    best_model = best_result["model"]

    print("\n========== BEST MRI-ALIGNED GFV MODEL ==========")
    print("Best config:", best_cfg)
    print("Best epoch:", best_state["epoch"])
    print("Selection score:", best_state["selection_score"])
    print("Validation MRI AUC:", best_result["val_metrics"]["mri_auc"])
    print("Validation MRI balanced accuracy:", best_result["val_metrics"]["mri_balanced_accuracy"])
    print("Validation alignment cosine:", best_result["val_metrics"]["alignment_cosine_mean"])
    print("Test MRI AUC:", best_result["test_metrics"]["mri_auc"])
    print("Test MRI balanced accuracy:", best_result["test_metrics"]["mri_balanced_accuracy"])
    print("Test alignment cosine:", best_result["test_metrics"]["alignment_cosine_mean"])
    print("===============================================\n")

    # Encode final outputs
    real_gfv_302_scaled = gfv_scaler.transform(real_gfv_302[gfv_cols].astype(float).values)
    mri_only_scaled = mfv_scaler.transform(mri_only[mfv_cols].astype(float).values)

    agfv_cols = [f"AGFV{j+1:03d}" for j in range(AGFV_DIM)]

    z_g_302, prob_g_302 = encode_batches(best_model, real_gfv_302_scaled, "gfv", device)
    z_m_paired, prob_m_paired = encode_batches(best_model, mfv_scaled, "mfv", device)
    z_m_mri_only, prob_m_mri_only = encode_batches(best_model, mri_only_scaled, "mfv", device)

    measured_out = pd.concat(
        [
            real_gfv_302[safe_meta_cols(real_gfv_302)].reset_index(drop=True),
            pd.DataFrame(z_g_302, columns=agfv_cols),
        ],
        axis=1,
    )
    measured_out.insert(0, "agfv_source", "measured_gfv_encoder")
    measured_out["aligned_AD_probability"] = prob_g_302

    paired_mri_out = pd.concat(
        [
            paired[safe_meta_cols(paired)].reset_index(drop=True),
            pd.DataFrame(z_m_paired, columns=agfv_cols),
        ],
        axis=1,
    )
    paired_mri_out.insert(0, "agfv_source", "mri_mfv_encoder")
    paired_mri_out["aligned_AD_probability"] = prob_m_paired

    mri_only_out = pd.concat(
        [
            mri_only[safe_meta_cols(mri_only)].reset_index(drop=True),
            pd.DataFrame(z_m_mri_only, columns=agfv_cols),
        ],
        axis=1,
    )
    mri_only_out.insert(0, "agfv_source", "mri_mfv_encoder")
    mri_only_out["aligned_AD_probability"] = prob_m_mri_only

    combined_out = pd.concat([measured_out, mri_only_out], axis=0, ignore_index=True)

    if measured_out.shape[0] != 302:
        raise ValueError(f"Expected 302 measured AGFV rows, got {measured_out.shape[0]}")
    if paired_mri_out.shape[0] != 289:
        raise ValueError(f"Expected 289 paired MRI AGFV rows, got {paired_mri_out.shape[0]}")
    if mri_only_out.shape[0] != 1206:
        raise ValueError(f"Expected 1206 MRI-only AGFV rows, got {mri_only_out.shape[0]}")
    if combined_out.shape[0] != 1508:
        raise ValueError(f"Expected 1508 combined AGFV rows, got {combined_out.shape[0]}")

    metrics_df = pd.DataFrame(all_metrics)
    history_df = pd.concat(all_histories, ignore_index=True)
    best_metrics_df = metrics_df[metrics_df["Config"] == best_cfg["name"]].copy()

    model_path = OUT_DIR / "final_aligned_gfv_model.pt"
    scaler_path = OUT_DIR / "final_aligned_gfv_scalers.pkl"
    config_path = OUT_DIR / "final_aligned_gfv_config.json"
    metrics_path = OUT_DIR / "final_aligned_gfv_metrics.csv"
    all_metrics_path = OUT_DIR / "aligned_gfv_all_config_metrics.csv"
    history_path = OUT_DIR / "aligned_gfv_training_history.csv"

    measured_out_path = OUT_DIR / "aligned_gfv_measured_ge_302_subjects.csv"
    paired_mri_out_path = OUT_DIR / "aligned_gfv_paired_mri_289_subjects.csv"
    mri_only_out_path = OUT_DIR / "aligned_gfv_mri_only_1206_subjects.csv"
    combined_out_path = OUT_DIR / "aligned_gfv_combined_1508_subjects.csv"

    table_path = TABLE_DIR / "Table_S15_Aligned_GFV_model_metrics.xlsx"

    torch.save(best_state, model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(
            {
                "gfv_scaler": gfv_scaler,
                "mfv_scaler": mfv_scaler,
                "gfv_cols": gfv_cols,
                "mfv_cols": mfv_cols,
                "agfv_cols": agfv_cols,
            },
            f,
        )

    final_config = {
        "model": "MRI-aligned molecular representation model",
        "seed": SEED,
        "best_config": best_cfg,
        "best_epoch": int(best_state["epoch"]),
        "selection_metric": best_state["selection_metric"],
        "selection_score": float(best_state["selection_score"]),
        "gfv_dim": GFV_DIM,
        "mfv_dim": MFV_DIM,
        "agfv_dim": AGFV_DIM,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": best_cfg.get("lr", LR),
        "weight_decay": best_cfg.get("weight_decay", WEIGHT_DECAY),
        "paired_subjects": int(paired.shape[0]),
        "measured_ge_subjects": int(real_gfv_302.shape[0]),
        "mri_only_subjects": int(mri_only.shape[0]),
        "combined_subjects": int(combined_out.shape[0]),
        "validation_mri_auc": float(best_result["val_metrics"]["mri_auc"]),
        "validation_mri_balanced_accuracy": float(best_result["val_metrics"]["mri_balanced_accuracy"]),
        "validation_alignment_cosine": float(best_result["val_metrics"]["alignment_cosine_mean"]),
        "test_mri_auc": float(best_result["test_metrics"]["mri_auc"]),
        "test_mri_balanced_accuracy": float(best_result["test_metrics"]["mri_balanced_accuracy"]),
        "test_alignment_cosine": float(best_result["test_metrics"]["alignment_cosine_mean"]),
    }

    with open(config_path, "w") as f:
        json.dump(final_config, f, indent=2)

    best_metrics_df.to_csv(metrics_path, index=False)
    metrics_df.to_csv(all_metrics_path, index=False)
    history_df.to_csv(history_path, index=False)

    measured_out.to_csv(measured_out_path, index=False)
    paired_mri_out.to_csv(paired_mri_out_path, index=False)
    mri_only_out.to_csv(mri_only_out_path, index=False)
    combined_out.to_csv(combined_out_path, index=False)

    with pd.ExcelWriter(table_path) as writer:
        best_metrics_df.to_excel(writer, sheet_name="Best_model_metrics", index=False)
        metrics_df.to_excel(writer, sheet_name="All_config_metrics", index=False)

    print("========== FINAL MRI-ALIGNED GFV BEST MODEL METRICS ==========")
    print(best_metrics_df.to_string(index=False))
    print("==============================================================\n")

    print("Saved:")
    print(model_path)
    print(scaler_path)
    print(config_path)
    print(metrics_path)
    print(all_metrics_path)
    print(history_path)
    print(measured_out_path)
    print(paired_mri_out_path)
    print(mri_only_out_path)
    print(combined_out_path)
    print(table_path)
    print("\nSTATUS: DONE")


if __name__ == "__main__":
    main()
