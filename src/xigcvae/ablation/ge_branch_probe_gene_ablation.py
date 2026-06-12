from __future__ import annotations

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

from src.xigcvae.models.gene_branch import GeneExpressionBranch


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

GE_FILE = PROJECT_ROOT / "data/gene_expression_data.csv"
META_FILE = PROJECT_ROOT / "data/Paired_data_metadata.csv"
MANIFEST_FILE = PROJECT_ROOT / "manifests/gene_branch_final_split_seed_42.csv"

MODEL_FILE = PROJECT_ROOT / "outputs/gene_branch/final_gene_branch_model.pt"
SCALER_FILE = PROJECT_ROOT / "outputs/gene_branch/final_gene_branch_scaler.pkl"
SELECTED_PROBE_FILE = PROJECT_ROOT / "outputs/gene_branch/final_selected_probe_list.csv"

OUT_DIR = PROJECT_ROOT / "outputs/ablation"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

PROBE_OUT = OUT_DIR / "ge_branch_probe_ablation.csv"
GENE_OUT = OUT_DIR / "ge_branch_gene_ablation.csv"
SUMMARY_OUT = OUT_DIR / "ge_branch_ablation_summary.txt"
TABLE_OUT = TABLE_DIR / "Table_S18_GE_branch_probe_gene_ablation.xlsx"


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


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


def load_expression_matrix():
    ge = pd.read_csv(GE_FILE, low_memory=False)

    first_col = ge.iloc[:, 0].astype(str).str.strip()
    matches = ge.index[first_col == "ProbeSet"].tolist()

    if not matches:
        raise ValueError("Could not find row labeled ProbeSet in the first column.")

    probeset_row = matches[0]

    expr = ge.iloc[probeset_row + 1:].copy()

    probe_col = expr.columns[0]
    locus_col = expr.columns[1]
    symbol_col = expr.columns[2]

    expr[probe_col] = expr[probe_col].astype(str).str.strip()

    return expr, probe_col, locus_col, symbol_col


def prepare_input():
    selected = pd.read_csv(SELECTED_PROBE_FILE)

    if "Probe ID / feature ID" not in selected.columns:
        raise ValueError("Selected probe list must contain 'Probe ID / feature ID'.")

    selected["Probe ID / feature ID"] = selected["Probe ID / feature ID"].astype(str)
    panel_probes = selected["Probe ID / feature ID"].tolist()

    expr, probe_col, locus_col, symbol_col = load_expression_matrix()

    meta = pd.read_csv(META_FILE)

    required = ["subject_id", "ge_sample_column", "diagnosis", "diagnosis_binary", "ge_column_found"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    meta = meta.copy()
    meta = meta[to_bool(meta["ge_column_found"])]
    meta = meta[meta["diagnosis"].astype(str).isin(["AD", "CN"])].copy()
    meta["diagnosis_binary"] = meta["diagnosis_binary"].astype(int)
    meta["subject_id"] = meta["subject_id"].astype(str)

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["subject_id"] = manifest["subject_id"].astype(str)

    meta = meta.merge(
        manifest[["subject_id", "split"]].drop_duplicates(),
        on="subject_id",
        how="left",
        suffixes=("", "_manifest"),
    )

    if "split_manifest" in meta.columns:
        meta["split"] = meta["split_manifest"]

    if "split" not in meta.columns or meta["split"].isna().any():
        raise ValueError("Could not assign split from manifest.")

    sample_cols = meta["ge_sample_column"].astype(str).tolist()
    missing_sample_cols = [c for c in sample_cols if c not in expr.columns]
    if missing_sample_cols:
        raise ValueError(f"Missing GE sample columns. Examples: {missing_sample_cols[:5]}")

    expr_panel = expr[expr[probe_col].astype(str).isin(panel_probes)].copy()
    expr_panel = expr_panel.drop_duplicates(subset=[probe_col], keep="first")
    expr_panel = expr_panel.set_index(probe_col)

    available_probes = [p for p in panel_probes if p in expr_panel.index]

    if len(available_probes) == 0:
        raise ValueError("No selected probes found in expression matrix.")

    expr_panel = expr_panel.loc[available_probes]

    x_df = expr_panel[sample_cols].T
    x_df.index = meta.index
    x_df = x_df.apply(pd.to_numeric, errors="coerce")

    selected_used = selected[
        selected["Probe ID / feature ID"].astype(str).isin(available_probes)
    ].copy()

    selected_used = selected_used.drop_duplicates(subset=["Probe ID / feature ID"], keep="first")
    selected_used = selected_used.set_index("Probe ID / feature ID").loc[available_probes].reset_index()

    return x_df, meta.reset_index(drop=True), selected_used, available_probes


@torch.no_grad()
def predict(model, x_np, device):
    model.eval()

    probs = []
    preds = []

    for start in range(0, x_np.shape[0], 256):
        end = min(start + 256, x_np.shape[0])
        xb = torch.tensor(x_np[start:end], dtype=torch.float32, device=device)
        logits, _ = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)
        probs.append(prob.cpu().numpy())
        preds.append(pred.cpu().numpy())

    return np.concatenate(probs), np.concatenate(preds)


def split_gene_symbols(value):
    if pd.isna(value):
        return ["Unknown"]

    genes = []
    for g in str(value).split("||"):
        g = g.strip()
        if g and g.lower() not in ["nan", "none", ""]:
            genes.append(g)

    return genes if genes else ["Unknown"]


def main():
    print("========== GE BRANCH PROBE/GENE ABLATION ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model: {MODEL_FILE}")
    print(f"Selected probes: {SELECTED_PROBE_FILE}")
    print("===================================================\n")

    for f in [GE_FILE, META_FILE, MANIFEST_FILE, MODEL_FILE, SCALER_FILE, SELECTED_PROBE_FILE]:
        if not f.exists():
            raise FileNotFoundError(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_df, meta, selected_used, available_probes = prepare_input()

    y = meta["diagnosis_binary"].astype(int).values
    split = meta["split"].astype(str).values

    with open(SCALER_FILE, "rb") as f:
        preprocess = pickle.load(f)

    x_scaled = preprocess.transform(x_df)

    ckpt = torch.load(MODEL_FILE, map_location=device)
    input_dim = int(ckpt.get("input_dim", x_scaled.shape[1]))
    gfv_dim = int(ckpt.get("gfv_dim", 128))
    dropout = float(ckpt.get("dropout", 0.60))

    model = GeneExpressionBranch(
        input_dim=input_dim,
        gfv_dim=gfv_dim,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if x_scaled.shape[1] != input_dim:
        raise ValueError(f"Input dimension mismatch: X={x_scaled.shape[1]}, model={input_dim}")

    baseline_prob, baseline_pred = predict(model, x_scaled, device)

    groups = {
        "Validation": np.where(split == "Validation")[0],
        "Test": np.where(split == "Test")[0],
        "Heldout": np.where(np.isin(split, ["Validation", "Test"]))[0],
    }

    baseline_metrics = {}
    for group_name, idx in groups.items():
        baseline_metrics[group_name] = compute_metrics(y[idx], baseline_prob[idx], baseline_pred[idx])

    rows = []

    print("Baseline metrics:")
    for group_name, m in baseline_metrics.items():
        print(
            f"{group_name}: AUC={m['auc']:.4f}, "
            f"BACC={m['balanced_accuracy']:.4f}, "
            f"ACC={m['accuracy']:.4f}"
        )
    print()

    n_features = x_scaled.shape[1]

    for j, probe_id in enumerate(available_probes):
        if (j + 1) % 100 == 0 or j == 0:
            print(f"Ablating probe {j + 1}/{n_features}: {probe_id}")

        x_ab = x_scaled.copy()

        # Because inputs are standardized by the training scaler,
        # setting one feature to 0 equals replacing it by the training-set mean.
        x_ab[:, j] = 0.0

        ab_prob, ab_pred = predict(model, x_ab, device)

        probe_meta = selected_used.iloc[j].to_dict()

        for group_name, idx in groups.items():
            ab_metrics = compute_metrics(y[idx], ab_prob[idx], ab_pred[idx])
            base = baseline_metrics[group_name]

            row = {
                "Split group": group_name,
                "Probe ID / feature ID": probe_id,
                "Feature index": int(j),
                "Gene symbol": probe_meta.get("Gene symbol", "Unknown"),
                "Genomic locus": probe_meta.get("Genomic locus", probe_meta.get("Locus", np.nan)),
                "Baseline AUC": base["auc"],
                "Ablated AUC": ab_metrics["auc"],
                "Delta AUC": base["auc"] - ab_metrics["auc"],
                "Baseline balanced accuracy": base["balanced_accuracy"],
                "Ablated balanced accuracy": ab_metrics["balanced_accuracy"],
                "Delta balanced accuracy": base["balanced_accuracy"] - ab_metrics["balanced_accuracy"],
                "Baseline accuracy": base["accuracy"],
                "Ablated accuracy": ab_metrics["accuracy"],
                "Delta accuracy": base["accuracy"] - ab_metrics["accuracy"],
                "Mean absolute probability change": float(np.mean(np.abs(baseline_prob[idx] - ab_prob[idx]))),
                "Mean signed probability change": float(np.mean(baseline_prob[idx] - ab_prob[idx])),
            }

            # Preserve useful annotation columns if present.
            for col in selected_used.columns:
                if col not in row and col != "Probe ID / feature ID":
                    row[col] = probe_meta.get(col, np.nan)

            rows.append(row)

    probe_df = pd.DataFrame(rows)

    # Rank heldout probes first.
    heldout_probe = probe_df[probe_df["Split group"] == "Heldout"].copy()
    heldout_probe["Probe ablation score"] = (
        heldout_probe["Delta AUC"].fillna(0)
        + heldout_probe["Delta balanced accuracy"].fillna(0)
        + 0.10 * heldout_probe["Mean absolute probability change"].fillna(0)
    )

    # Gene-level aggregation by exploding gene symbols.
    gene_rows = []
    for _, r in heldout_probe.iterrows():
        for gene in split_gene_symbols(r.get("Gene symbol", "Unknown")):
            tmp = r.to_dict()
            tmp["Gene"] = gene
            gene_rows.append(tmp)

    gene_long = pd.DataFrame(gene_rows)

    gene_df = (
        gene_long.groupby("Gene")
        .agg(
            Probe_count=("Probe ID / feature ID", "nunique"),
            Max_delta_AUC=("Delta AUC", "max"),
            Mean_delta_AUC=("Delta AUC", "mean"),
            Max_delta_balanced_accuracy=("Delta balanced accuracy", "max"),
            Mean_delta_balanced_accuracy=("Delta balanced accuracy", "mean"),
            Max_probability_change=("Mean absolute probability change", "max"),
            Mean_probability_change=("Mean absolute probability change", "mean"),
            Max_probe_ablation_score=("Probe ablation score", "max"),
            Mean_probe_ablation_score=("Probe ablation score", "mean"),
        )
        .reset_index()
        .sort_values(
            ["Max_probe_ablation_score", "Max_delta_AUC", "Max_delta_balanced_accuracy"],
            ascending=False,
        )
    )

    probe_df.to_csv(PROBE_OUT, index=False)
    gene_df.to_csv(GENE_OUT, index=False)

    with pd.ExcelWriter(TABLE_OUT) as writer:
        heldout_probe.sort_values("Probe ablation score", ascending=False).to_excel(
            writer,
            sheet_name="Heldout_probe_ablation",
            index=False,
        )
        gene_df.to_excel(writer, sheet_name="Heldout_gene_ablation", index=False)
        probe_df.to_excel(writer, sheet_name="All_split_probe_ablation", index=False)

    summary = []
    summary.append("========== GE BRANCH PROBE/GENE ABLATION ==========")
    summary.append(f"Subjects: {len(meta)}")
    summary.append(f"Features/probes: {n_features}")
    summary.append("")
    summary.append("Baseline metrics:")
    for group_name, m in baseline_metrics.items():
        summary.append(
            f"{group_name}: AUC={m['auc']:.6f}, "
            f"Balanced accuracy={m['balanced_accuracy']:.6f}, "
            f"Accuracy={m['accuracy']:.6f}"
        )
    summary.append("")
    summary.append("Top 20 heldout probes by ablation score:")
    top_probes = heldout_probe.sort_values("Probe ablation score", ascending=False).head(20)
    summary.append(
        top_probes[
            [
                "Probe ID / feature ID",
                "Gene symbol",
                "Delta AUC",
                "Delta balanced accuracy",
                "Mean absolute probability change",
                "Probe ablation score",
            ]
        ].to_string(index=False)
    )
    summary.append("")
    summary.append("Top 20 heldout genes by ablation score:")
    summary.append(
        gene_df[
            [
                "Gene",
                "Probe_count",
                "Max_delta_AUC",
                "Max_delta_balanced_accuracy",
                "Max_probability_change",
                "Max_probe_ablation_score",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Saved:")
    summary.append(str(PROBE_OUT))
    summary.append(str(GENE_OUT))
    summary.append(str(TABLE_OUT))
    summary.append("STATUS: DONE")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print()
    print("\n".join(summary))
    print("====================================================")


if __name__ == "__main__":
    main()
