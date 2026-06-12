from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.xigcvae.models.agfv_nn_classifier import AGFVNNClassifier


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

AGFV_FILE = PROJECT_ROOT / "outputs/aligned_gfv/aligned_gfv_combined_1508_subjects.csv"

NN_MODEL_FILE = PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_model.pt"
NN_SCALER_FILE = PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_scaler.pkl"
NN_CONFIG_FILE = PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_config.json"

OUT_DIR = PROJECT_ROOT / "outputs/ablation"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_OUT = OUT_DIR / "agfv_nn_feature_ablation.csv"
SUMMARY_OUT = OUT_DIR / "agfv_nn_feature_ablation_summary.txt"
TABLE_OUT = TABLE_DIR / "Table_S19_AGFV_NN_feature_ablation.xlsx"


def get_agfv_cols(df: pd.DataFrame):
    pat = re.compile(r"^AGFV\d+$", re.IGNORECASE)
    cols = [c for c in df.columns if pat.match(c)]
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0]))


def get_label(df: pd.DataFrame):
    if "diagnosis_binary" in df.columns:
        return df["diagnosis_binary"].astype(int).values

    if "diagnosis" in df.columns:
        return df["diagnosis"].map({"CN": 0, "AD": 1}).astype(int).values

    raise ValueError("Input file must contain diagnosis_binary or diagnosis.")


def safe_auc(y_true, prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, prob))


def compute_metrics(y_true, prob, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "subjects_n": int(len(y_true)),
        "auc": safe_auc(y_true, prob),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "ad_sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "cn_specificity": float(specificity),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def predict(model, x_np, device):
    model.eval()

    probs = []
    preds = []

    for start in range(0, x_np.shape[0], 512):
        end = min(start + 512, x_np.shape[0])
        xb = torch.tensor(x_np[start:end], dtype=torch.float32, device=device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)

        probs.append(prob.cpu().numpy())
        preds.append(pred.cpu().numpy())

    return np.concatenate(probs), np.concatenate(preds)


def load_model(device):
    ckpt = torch.load(NN_MODEL_FILE, map_location=device)

    cfg = ckpt["config"]
    hidden_dims = tuple(cfg["hidden_dims"])
    dropout = float(cfg["dropout"])

    model = AGFVNNClassifier(
        input_dim=128,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def main():
    print("========== AGFV NN FEATURE ABLATION ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"AGFV file: {AGFV_FILE}")
    print(f"NN model: {NN_MODEL_FILE}")
    print("==============================================\n")

    for f in [AGFV_FILE, NN_MODEL_FILE, NN_SCALER_FILE, NN_CONFIG_FILE]:
        if not f.exists():
            raise FileNotFoundError(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(AGFV_FILE)
    agfv_cols = get_agfv_cols(df)

    if len(agfv_cols) != 128:
        raise ValueError(f"Expected 128 AGFV columns, found {len(agfv_cols)}")

    if "split" not in df.columns:
        raise ValueError("AGFV file must contain split column.")

    if "subject_id" not in df.columns:
        raise ValueError("AGFV file must contain subject_id column.")

    if df["subject_id"].duplicated().sum() > 0:
        raise ValueError("Duplicate subject_id found.")

    y = get_label(df)

    with open(NN_SCALER_FILE, "rb") as f:
        scaler_obj = pickle.load(f)

    scaler = scaler_obj["scaler"]
    saved_cols = scaler_obj["agfv_cols"]

    if list(saved_cols) != list(agfv_cols):
        raise ValueError("AGFV columns in scaler do not match AGFV file.")

    x_raw = df[agfv_cols].astype(float).values
    x_scaled = scaler.transform(x_raw).astype(np.float32)

    model, ckpt = load_model(device)

    baseline_prob, baseline_pred = predict(model, x_scaled, device)

    split = df["split"].astype(str).values

    groups = {
        "Train": np.where(split == "Train")[0],
        "Validation": np.where(split == "Validation")[0],
        "Test": np.where(split == "Test")[0],
        "Heldout": np.where(np.isin(split, ["Validation", "Test"]))[0],
        "All": np.arange(df.shape[0]),
    }

    baseline_metrics = {}
    for group_name, idx in groups.items():
        baseline_metrics[group_name] = compute_metrics(
            y[idx],
            baseline_prob[idx],
            baseline_pred[idx],
        )

    print("Baseline metrics:")
    for group_name in ["Train", "Validation", "Test", "Heldout", "All"]:
        m = baseline_metrics[group_name]
        print(
            f"{group_name}: "
            f"AUC={m['auc']:.4f}, "
            f"BACC={m['balanced_accuracy']:.4f}, "
            f"ACC={m['accuracy']:.4f}, "
            f"F1={m['f1']:.4f}"
        )
    print()

    rows = []

    for j, feature_name in enumerate(agfv_cols):
        if j == 0 or (j + 1) % 20 == 0:
            print(f"Ablating AGFV feature {j + 1}/128: {feature_name}")

        x_ab = x_scaled.copy()

        # Because features are standardized by the training scaler,
        # setting one feature to 0 equals replacing it by the training-set mean.
        x_ab[:, j] = 0.0

        ab_prob, ab_pred = predict(model, x_ab, device)

        for group_name, idx in groups.items():
            base = baseline_metrics[group_name]
            ab_metrics = compute_metrics(y[idx], ab_prob[idx], ab_pred[idx])

            mean_abs_prob_change = float(np.mean(np.abs(baseline_prob[idx] - ab_prob[idx])))
            mean_signed_prob_change = float(np.mean(baseline_prob[idx] - ab_prob[idx]))

            row = {
                "Split group": group_name,
                "AGFV feature": feature_name,
                "Feature index": int(j),
                "Baseline AUC": base["auc"],
                "Ablated AUC": ab_metrics["auc"],
                "Delta AUC": base["auc"] - ab_metrics["auc"],
                "Baseline balanced accuracy": base["balanced_accuracy"],
                "Ablated balanced accuracy": ab_metrics["balanced_accuracy"],
                "Delta balanced accuracy": base["balanced_accuracy"] - ab_metrics["balanced_accuracy"],
                "Baseline accuracy": base["accuracy"],
                "Ablated accuracy": ab_metrics["accuracy"],
                "Delta accuracy": base["accuracy"] - ab_metrics["accuracy"],
                "Baseline F1": base["f1"],
                "Ablated F1": ab_metrics["f1"],
                "Delta F1": base["f1"] - ab_metrics["f1"],
                "Mean absolute probability change": mean_abs_prob_change,
                "Mean signed probability change": mean_signed_prob_change,
            }

            rows.append(row)

    out = pd.DataFrame(rows)

    out["Feature ablation score"] = (
        out["Delta AUC"].fillna(0)
        + out["Delta balanced accuracy"].fillna(0)
        + 0.10 * out["Mean absolute probability change"].fillna(0)
    )

    heldout = out[out["Split group"] == "Heldout"].copy()
    validation = out[out["Split group"] == "Validation"].copy()
    test = out[out["Split group"] == "Test"].copy()

    heldout = heldout.sort_values(
        ["Feature ablation score", "Delta AUC", "Delta balanced accuracy"],
        ascending=False,
    )
    validation = validation.sort_values(
        ["Feature ablation score", "Delta AUC", "Delta balanced accuracy"],
        ascending=False,
    )
    test = test.sort_values(
        ["Feature ablation score", "Delta AUC", "Delta balanced accuracy"],
        ascending=False,
    )

    out.to_csv(FEATURE_OUT, index=False)

    with pd.ExcelWriter(TABLE_OUT) as writer:
        heldout.to_excel(writer, sheet_name="Heldout_AGFV_ablation", index=False)
        validation.to_excel(writer, sheet_name="Validation_AGFV_ablation", index=False)
        test.to_excel(writer, sheet_name="Test_AGFV_ablation", index=False)
        out.to_excel(writer, sheet_name="All_split_AGFV_ablation", index=False)

    summary = []
    summary.append("========== AGFV NN FEATURE ABLATION ==========")
    summary.append(f"Subjects: {df.shape[0]}")
    summary.append(f"AGFV features: {len(agfv_cols)}")
    summary.append(f"Best NN config: {ckpt['config']['name']}")
    summary.append(f"Best NN epoch: {ckpt['best_epoch']}")
    summary.append("")
    summary.append("Baseline metrics:")
    for group_name in ["Train", "Validation", "Test", "Heldout", "All"]:
        m = baseline_metrics[group_name]
        summary.append(
            f"{group_name}: AUC={m['auc']:.6f}, "
            f"Balanced accuracy={m['balanced_accuracy']:.6f}, "
            f"Accuracy={m['accuracy']:.6f}, "
            f"F1={m['f1']:.6f}"
        )

    summary.append("")
    summary.append("Top 25 AGFV features by heldout ablation score:")
    summary.append(
        heldout[
            [
                "AGFV feature",
                "Delta AUC",
                "Delta balanced accuracy",
                "Delta accuracy",
                "Delta F1",
                "Mean absolute probability change",
                "Feature ablation score",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )

    summary.append("")
    summary.append("Saved:")
    summary.append(str(FEATURE_OUT))
    summary.append(str(TABLE_OUT))
    summary.append("STATUS: DONE")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print()
    print("\n".join(summary))
    print("===============================================")


if __name__ == "__main__":
    main()
