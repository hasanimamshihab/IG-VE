from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

FILES = {
    "GE branch classifier": PROJECT_ROOT / "outputs/gene_branch/final_gene_branch_metrics.csv",
    "MRI branch classifier": PROJECT_ROOT / "outputs/mri_branch/final_mri_branch_metrics.csv",
    "AGFV logistic classifier": PROJECT_ROOT / "outputs/final_agfv_classifier/final_agfv_classifier_metrics.csv",
    "AGFV NN classifier": PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_metrics.csv",
}

OUT_DIR = PROJECT_ROOT / "outputs/tables/main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "Table_2_Model_performance_comparison.csv"
OUT_XLSX = OUT_DIR / "Table_2_Model_performance_comparison.xlsx"


def pick_col(df: pd.DataFrame, candidates: list[str]):
    normalized = {c.lower().replace("_", " ").replace("-", " ").strip(): c for c in df.columns}

    for cand in candidates:
        key = cand.lower().replace("_", " ").replace("-", " ").strip()
        if key in normalized:
            return normalized[key]

    return None


def get_value(row, col):
    if col is None:
        return np.nan
    return row[col]


def standardize_model_metrics(model_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "Subgroup" in df.columns:
        df = df[df["Subgroup"].astype(str) == "Overall"].copy()

    split_col = pick_col(df, ["Split", "split"])

    auc_col = pick_col(
        df,
        [
            "AUC",
            "auc",
            "MRI-derived AGFV AUC",
            "Validation AUC",
            "Test AUC",
        ],
    )

    bacc_col = pick_col(
        df,
        [
            "Balanced accuracy",
            "balanced_accuracy",
            "balanced acc",
            "MRI-derived AGFV balanced accuracy",
            "Aux balanced acc",
            "Test balanced acc",
        ],
    )

    acc_col = pick_col(
        df,
        [
            "Accuracy",
            "accuracy",
            "MRI-derived AGFV accuracy",
        ],
    )

    sens_col = pick_col(
        df,
        [
            "AD sensitivity",
            "AD sensitivity/recall",
            "recall",
            "recall_ad_sensitivity",
            "MRI-derived AGFV AD sensitivity",
            "MRI-derived AGFV recall_ad_sensitivity",
            "Sensitivity",
        ],
    )

    spec_col = pick_col(
        df,
        [
            "CN specificity",
            "Specificity",
            "cn_specificity",
            "MRI-derived AGFV CN specificity",
        ],
    )

    f1_col = pick_col(
        df,
        [
            "F1",
            "f1",
            "MRI-derived AGFV F1",
        ],
    )

    rows = []

    for _, r in df.iterrows():
        split = get_value(r, split_col)
        if split not in ["Train", "Validation", "Test"]:
            continue

        rows.append(
            {
                "Model": model_name,
                "Split": split,
                "AUC": get_value(r, auc_col),
                "Balanced accuracy": get_value(r, bacc_col),
                "Accuracy": get_value(r, acc_col),
                "AD sensitivity": get_value(r, sens_col),
                "CN specificity": get_value(r, spec_col),
                "F1": get_value(r, f1_col),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("========== BUILD MODEL PERFORMANCE COMPARISON ==========")

    all_rows = []
    for model_name, path in FILES.items():
        print(f"Reading {model_name}: {path}")
        tmp = standardize_model_metrics(model_name, path)
        all_rows.append(tmp)

    out = pd.concat(all_rows, ignore_index=True)

    model_order = {
        "GE branch classifier": 0,
        "MRI branch classifier": 1,
        "AGFV logistic classifier": 2,
        "AGFV NN classifier": 3,
    }
    split_order = {"Train": 0, "Validation": 1, "Test": 2}

    out["model_order"] = out["Model"].map(model_order)
    out["split_order"] = out["Split"].map(split_order)
    out = out.sort_values(["model_order", "split_order"]).drop(columns=["model_order", "split_order"])

    numeric_cols = ["AUC", "Balanced accuracy", "Accuracy", "AD sensitivity", "CN specificity", "F1"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out.to_csv(OUT_CSV, index=False)

    with pd.ExcelWriter(OUT_XLSX) as writer:
        out.to_excel(writer, sheet_name="Model_comparison", index=False)

    print()
    print(out.to_string(index=False))
    print()
    print("Saved:")
    print(OUT_CSV)
    print(OUT_XLSX)
    print("STATUS: DONE")


if __name__ == "__main__":
    main()
