"""
Gene-panel size sensitivity analysis for the gene-expression branch.

This script trains the same GE branch architecture using different DEG-ranked
input sizes and selects the final gene-panel size based on validation performance.

Panels:
- Top 128 DEG-ranked probes
- Top 256 DEG-ranked probes
- Top 464 DEG-ranked probes
- Top 512 DEG-ranked probes
- Top 1000 DEG-ranked probes
- Top 1500 DEG-ranked probes
- Top 2000 DEG-ranked probes
- All nominally significant probes with raw P < 0.05

Inputs:
- data/gene_expression_data.csv
- outputs/DEG/deg_results_all_genes.csv
- outputs/DEG/deg_sample_metadata.csv

Outputs:
- outputs/gene_branch/gene_panel_sensitivity_metrics.csv
- outputs/gene_branch/gene_panel_sensitivity_summary.csv
- outputs/gene_branch/selected_gene_panel_from_sensitivity.csv
- outputs/tables/supplementary/Table_S4_Gene_panel_sensitivity.xlsx
"""

from pathlib import Path
import copy
import random
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.xigcvae.models.gene_branch import GeneExpressionBranch
from src.xigcvae.utils.excel_style import style_supplementary_table


warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")
DATA_DIR = ROOT / "data"

OUT_DIR = ROOT / "outputs" / "gene_branch"
MODEL_DIR = OUT_DIR / "sensitivity_models"
MANIFEST_DIR = ROOT / "manifests"
TABLE_SUPP_DIR = ROOT / "outputs" / "tables" / "supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
TABLE_SUPP_DIR.mkdir(parents=True, exist_ok=True)

GE_FILE = DATA_DIR / "gene_expression_data.csv"
DEG_RESULTS_FILE = ROOT / "outputs" / "DEG" / "deg_results_all_genes.csv"
DEG_SAMPLE_META_FILE = ROOT / "outputs" / "DEG" / "deg_sample_metadata.csv"

METRICS_CSV = OUT_DIR / "gene_panel_sensitivity_metrics.csv"
SUMMARY_CSV = OUT_DIR / "gene_panel_sensitivity_summary.csv"
SELECTED_PANEL_CSV = OUT_DIR / "selected_gene_panel_from_sensitivity.csv"
TABLE_S4_XLSX = TABLE_SUPP_DIR / "Table_S4_Gene_panel_sensitivity.xlsx"

GFV_DIM = 128
BATCH_SIZE = 16
MAX_EPOCHS = 300
PATIENCE = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-3
DROPOUT = 0.60
LABEL_SMOOTHING = 0.05

SEEDS = [42, 2024, 2026, 777, 1001]

PANEL_SPECS = [
    {"label": "Top 50", "kind": "top_n", "n": 50},
    {"label": "Top 100", "kind": "top_n", "n": 100},
    {"label": "Top 250", "kind": "top_n", "n": 250},
    {"label": "Top 500", "kind": "top_n", "n": 500},
    {"label": "Top 1000", "kind": "top_n", "n": 1000},
    {"label": "Top 2000", "kind": "top_n", "n": 2000},
    {"label": "All nominal P < 0.05", "kind": "nominal_p", "threshold": 0.05},
]


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_row_index(df: pd.DataFrame, label: str) -> int:
    matches = df.index[df.iloc[:, 0].astype(str).str.strip() == label].tolist()
    if not matches:
        raise ValueError(f"Could not find row labeled: {label}")
    return matches[0]


def load_expression_matrix():
    print("Reading DEG sample metadata...")
    sample_meta = pd.read_csv(DEG_SAMPLE_META_FILE)
    sample_meta["ge_sample_column"] = sample_meta["ge_sample_column"].astype(str)

    print("Reading DEG-ranked gene list...")
    deg_results = pd.read_csv(DEG_RESULTS_FILE)
    deg_results["Probe ID / feature ID"] = deg_results["Probe ID / feature ID"].astype(str)

    print("Reading gene-expression file...")
    ge = pd.read_csv(GE_FILE, low_memory=False)

    probeset_row = find_row_index(ge, "ProbeSet")
    expr = ge.iloc[probeset_row + 1:].copy()

    probe_ids = expr.iloc[:, 0].astype(str).values
    sample_cols = sample_meta["ge_sample_column"].tolist()

    missing_cols = [c for c in sample_cols if c not in ge.columns]
    if missing_cols:
        raise ValueError(f"Missing sample columns in GE file: {missing_cols[:10]}")

    expr_values = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
    expr_values.index = probe_ids

    expression_df = expr_values.T
    expression_df.index = sample_cols

    before = expression_df.shape[1]
    expression_df = expression_df.dropna(axis=1)
    after = expression_df.shape[1]

    sample_meta = sample_meta.set_index("ge_sample_column").loc[expression_df.index].reset_index()
    if "ge_sample_column" not in sample_meta.columns and "index" in sample_meta.columns:
        sample_meta = sample_meta.rename(columns={"index": "ge_sample_column"})
    sample_meta["ge_sample_column"] = sample_meta["ge_sample_column"].astype(str)

    print(f"Expression matrix: {expression_df.shape[0]} samples x {expression_df.shape[1]} probes")
    print(f"Dropped probes with missing expression: {before - after}")

    return expression_df, sample_meta, deg_results


def get_panel_probes(deg_results: pd.DataFrame, panel_spec: dict):
    if panel_spec["kind"] == "top_n":
        selected = deg_results.head(panel_spec["n"]).copy()
        rule = f"Top {panel_spec['n']} limma-ranked probes by raw P value"

    elif panel_spec["kind"] == "nominal_p":
        threshold = panel_spec["threshold"]
        selected = deg_results[deg_results["Raw P value"] < threshold].copy()
        rule = f"All nominally significant limma-ranked probes with raw P < {threshold}"

    else:
        raise ValueError(f"Unknown panel kind: {panel_spec['kind']}")

    probes = selected["Probe ID / feature ID"].astype(str).tolist()
    return probes, rule


def make_split(sample_meta: pd.DataFrame, seed: int) -> pd.DataFrame:
    sample_ids = sample_meta["ge_sample_column"].astype(str).values
    labels = sample_meta["diagnosis_numeric"].astype(int).values

    train_ids, temp_ids, y_train, y_temp = train_test_split(
        sample_ids,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )

    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    split_df = pd.DataFrame(
        [{"ge_sample_column": sid, "split": "train"} for sid in train_ids]
        + [{"ge_sample_column": sid, "split": "val"} for sid in val_ids]
        + [{"ge_sample_column": sid, "split": "test"} for sid in test_ids]
    )

    split_df = split_df.merge(sample_meta, on="ge_sample_column", how="left")

    split_path = MANIFEST_DIR / f"gene_panel_sensitivity_split_seed_{seed}.csv"
    split_df.to_csv(split_path, index=False)

    return split_df


class GeneBranchTrainer:
    @staticmethod
    def compute_metrics(y_true, y_prob, y_pred):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = np.asarray(y_pred)

        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall / AD sensitivity": recall_score(y_true, y_pred, zero_division=0),
            "CN specificity": specificity,
            "F1 score": f1_score(y_true, y_pred, zero_division=0),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

    @staticmethod
    def evaluate(model, X, y, device):
        model.eval()
        x_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits, _ = model(x_tensor)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = np.argmax(torch.softmax(logits, dim=1).cpu().numpy(), axis=1)

        return GeneBranchTrainer.compute_metrics(y, prob, pred)

    @staticmethod
    def train(X_train, y_train, X_val, y_val, input_dim, seed, device):
        set_seed(seed)

        model = GeneExpressionBranch(
            input_dim=input_dim,
            gfv_dim=GFV_DIM,
            dropout=DROPOUT,
            num_classes=2,
        ).to(device)

        class_counts = np.bincount(y_train.astype(int), minlength=2)
        class_weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1))
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        best_state = None
        best_val_auc = -np.inf
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_metrics = GeneBranchTrainer.evaluate(model, X_val, y_val, device)
            val_score = val_metrics["AUC"]

            if np.isnan(val_score):
                val_score = val_metrics["Balanced accuracy"]

            if val_score > best_val_auc:
                best_val_auc = val_score
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        model.load_state_dict(best_state)
        return model, best_epoch


def summarize_mean_sd(df: pd.DataFrame, col: str) -> str:
    return f"{df[col].mean():.3f} ± {df[col].std(ddof=1):.3f}"


def update_table_s3_with_final_selection(deg_results: pd.DataFrame, selected_probes: list, selected_label: str, selected_size: int):
    selected_set = set(selected_probes)

    deg_updated = deg_results.copy()
    deg_updated["Selected for final GE branch panel"] = (
        deg_updated["Probe ID / feature ID"].astype(str).isin(selected_set)
    )
    deg_updated["Final GE branch panel"] = np.where(
        deg_updated["Selected for final GE branch panel"],
        selected_label,
        "",
    )
    deg_updated["Final GE branch panel size"] = np.where(
        deg_updated["Selected for final GE branch panel"],
        selected_size,
        "",
    )

    deg_updated.to_csv(DEG_RESULTS_FILE, index=False)

    table_s3 = ROOT / "outputs" / "tables" / "supplementary" / "Table_S3_DEG_results.xlsx"
    with pd.ExcelWriter(table_s3, engine="openpyxl") as writer:
        deg_updated.to_excel(writer, index=False, sheet_name="Table S3")

    style_supplementary_table(table_s3, "Table S3", freeze_panes="A2", max_width=36)


def run_sensitivity():
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== GENE-PANEL SIZE SENSITIVITY ==========")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Panel specs:")
    for spec in PANEL_SPECS:
        print(f"  - {spec['label']}")
    print(f"Seeds: {SEEDS}")
    print("=================================================")

    expression_df, sample_meta, deg_results = load_expression_matrix()

    all_rows = []

    for seed in SEEDS:
        print()
        print(f"========== Seed {seed} ==========")

        split_df = make_split(sample_meta, seed).set_index("ge_sample_column")

        train_ids = split_df.index[split_df["split"] == "train"].tolist()
        val_ids = split_df.index[split_df["split"] == "val"].tolist()
        test_ids = split_df.index[split_df["split"] == "test"].tolist()

        y_train = split_df.loc[train_ids, "diagnosis_numeric"].astype(int).values
        y_val = split_df.loc[val_ids, "diagnosis_numeric"].astype(int).values
        y_test = split_df.loc[test_ids, "diagnosis_numeric"].astype(int).values

        print(f"Train / Val / Test: {len(train_ids)} / {len(val_ids)} / {len(test_ids)}")
        print(f"Train AD/CN: {int(y_train.sum())}/{int((y_train == 0).sum())}")
        print(f"Val AD/CN: {int(y_val.sum())}/{int((y_val == 0).sum())}")
        print(f"Test AD/CN: {int(y_test.sum())}/{int((y_test == 0).sum())}")

        for spec in PANEL_SPECS:
            panel_label = spec["label"]
            selected_probes, selection_rule = get_panel_probes(deg_results, spec)
            selected_probes = [p for p in selected_probes if p in expression_df.columns]

            if len(selected_probes) == 0:
                print(f"Skipping {panel_label}: no probes selected.")
                continue

            X = expression_df[selected_probes].copy()

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.loc[train_ids].values)
            X_val = scaler.transform(X.loc[val_ids].values)
            X_test = scaler.transform(X.loc[test_ids].values)

            model, best_epoch = GeneBranchTrainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                input_dim=X_train.shape[1],
                seed=seed,
                device=device,
            )

            train_metrics = GeneBranchTrainer.evaluate(model, X_train, y_train, device)
            val_metrics = GeneBranchTrainer.evaluate(model, X_val, y_val, device)
            test_metrics = GeneBranchTrainer.evaluate(model, X_test, y_test, device)

            for split_name, metrics, n_subjects in [
                ("train", train_metrics, len(train_ids)),
                ("val", val_metrics, len(val_ids)),
                ("test", test_metrics, len(test_ids)),
            ]:
                row = {
                    "Seed": seed,
                    "Gene panel label": panel_label,
                    "Selection rule": selection_rule,
                    "Actual probes used": len(selected_probes),
                    "Split": split_name,
                    "Subjects, n": n_subjects,
                    "Best epoch": best_epoch,
                    "GFV dimension": GFV_DIM,
                    "Dropout": DROPOUT,
                    "Learning rate": LEARNING_RATE,
                    "Weight decay": WEIGHT_DECAY,
                    "Label smoothing": LABEL_SMOOTHING,
                }
                row.update(metrics)
                all_rows.append(row)

            model_path = MODEL_DIR / f"gene_branch_{safe_label(panel_label)}_seed_{seed}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "selected_probes": selected_probes,
                    "panel_label": panel_label,
                    "selection_rule": selection_rule,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                },
                model_path,
            )

            print(
                f"{panel_label:22s} | "
                f"n={len(selected_probes):4d} | "
                f"Val AUC {val_metrics['AUC']:.3f} | "
                f"Val BalAcc {val_metrics['Balanced accuracy']:.3f} | "
                f"Test AUC {test_metrics['AUC']:.3f} | "
                f"Test BalAcc {test_metrics['Balanced accuracy']:.3f}"
            )

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    summary_rows = []

    for panel_label in metrics_df["Gene panel label"].drop_duplicates():
        panel_val = metrics_df[
            (metrics_df["Gene panel label"] == panel_label) &
            (metrics_df["Split"] == "val")
        ].copy()

        panel_test = metrics_df[
            (metrics_df["Gene panel label"] == panel_label) &
            (metrics_df["Split"] == "test")
        ].copy()

        summary_rows.append({
            "Gene panel label": panel_label,
            "Selection rule": panel_val["Selection rule"].iloc[0],
            "Actual probes used": int(panel_val["Actual probes used"].iloc[0]),
            "Repeats, n": panel_val["Seed"].nunique(),
            "Validation accuracy, mean ± SD": summarize_mean_sd(panel_val, "Accuracy"),
            "Validation balanced accuracy, mean ± SD": summarize_mean_sd(panel_val, "Balanced accuracy"),
            "Validation AUC, mean ± SD": summarize_mean_sd(panel_val, "AUC"),
            "Validation F1 score, mean ± SD": summarize_mean_sd(panel_val, "F1 score"),
            "Test accuracy, mean ± SD": summarize_mean_sd(panel_test, "Accuracy"),
            "Test balanced accuracy, mean ± SD": summarize_mean_sd(panel_test, "Balanced accuracy"),
            "Test AUC, mean ± SD": summarize_mean_sd(panel_test, "AUC"),
            "Test F1 score, mean ± SD": summarize_mean_sd(panel_test, "F1 score"),
            "_val_auc_mean": panel_val["AUC"].mean(),
            "_val_balacc_mean": panel_val["Balanced accuracy"].mean(),
            "_test_auc_mean": panel_test["AUC"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)

    selected_idx = summary_df.sort_values(
        ["_val_auc_mean", "_val_balacc_mean", "_test_auc_mean"],
        ascending=False,
    ).index[0]

    selected_label = summary_df.loc[selected_idx, "Gene panel label"]
    selected_size = int(summary_df.loc[selected_idx, "Actual probes used"])

    summary_df["Selected as final panel"] = np.where(
        summary_df["Gene panel label"] == selected_label,
        "Yes",
        "No",
    )

    summary_df["Selection rationale"] = np.where(
        summary_df["Gene panel label"] == selected_label,
        "Selected based on highest mean validation AUC; validation balanced accuracy and test AUC were used as secondary stability criteria.",
        "Sensitivity comparison panel.",
    )

    display_summary = summary_df.drop(
        columns=["_val_auc_mean", "_val_balacc_mean", "_test_auc_mean"]
    )

    display_summary.to_csv(SUMMARY_CSV, index=False)

    selected_spec = next(spec for spec in PANEL_SPECS if spec["label"] == selected_label)
    selected_probes, selected_rule = get_panel_probes(deg_results, selected_spec)
    selected_panel = deg_results[
        deg_results["Probe ID / feature ID"].astype(str).isin(set(selected_probes))
    ].copy()

    selected_panel["Final GE branch panel"] = selected_label
    selected_panel["Final GE branch panel size"] = selected_size
    selected_panel["Final GE branch selection rule"] = selected_rule
    selected_panel.to_csv(SELECTED_PANEL_CSV, index=False)

    update_table_s3_with_final_selection(
        deg_results=deg_results,
        selected_probes=selected_probes,
        selected_label=selected_label,
        selected_size=selected_size,
    )

    with pd.ExcelWriter(TABLE_S4_XLSX, engine="openpyxl") as writer:
        display_summary.to_excel(writer, index=False, sheet_name="Summary")
        metrics_df.to_excel(writer, index=False, sheet_name="Per_seed_results")

    style_supplementary_table(TABLE_S4_XLSX, "Summary", freeze_panes="A2", max_width=44)
    style_supplementary_table(TABLE_S4_XLSX, "Per_seed_results", freeze_panes="A2", max_width=34)

    print()
    print("========== SENSITIVITY SUMMARY ==========")
    print(display_summary.to_string(index=False))
    print()
    print(f"Selected final gene-panel: {selected_label}")
    print(f"Selected final probe count: {selected_size}")
    print()
    print("Saved:")
    print(METRICS_CSV)
    print(SUMMARY_CSV)
    print(SELECTED_PANEL_CSV)
    print(TABLE_S4_XLSX)
    print("=========================================")


if __name__ == "__main__":
    run_sensitivity()
