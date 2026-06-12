from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

OUT_DIR = PROJECT_ROOT / "outputs/aligned_gfv"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

MODEL_FILE = OUT_DIR / "final_aligned_gfv_model.pt"
SCALER_FILE = OUT_DIR / "final_aligned_gfv_scalers.pkl"
CONFIG_FILE = OUT_DIR / "final_aligned_gfv_config.json"
METRICS_FILE = OUT_DIR / "final_aligned_gfv_metrics.csv"
ALL_METRICS_FILE = OUT_DIR / "aligned_gfv_all_config_metrics.csv"
HISTORY_FILE = OUT_DIR / "aligned_gfv_training_history.csv"
MEASURED_FILE = OUT_DIR / "aligned_gfv_measured_ge_302_subjects.csv"
PAIRED_MRI_FILE = OUT_DIR / "aligned_gfv_paired_mri_289_subjects.csv"
MRI_ONLY_FILE = OUT_DIR / "aligned_gfv_mri_only_1206_subjects.csv"
COMBINED_FILE = OUT_DIR / "aligned_gfv_combined_1508_subjects.csv"
TABLE_FILE = TABLE_DIR / "Table_S15_Aligned_GFV_model_metrics.xlsx"

SUMMARY_FILE = OUT_DIR / "check_aligned_gfv_outputs_summary.txt"


def count_cols(df: pd.DataFrame, prefix: str):
    pat = re.compile(rf"^{prefix}\d+$", re.IGNORECASE)
    return sorted([c for c in df.columns if pat.match(c)])


def check(cond: bool, msg: str, errors: list[str], passes: list[str]):
    if cond:
        passes.append(msg)
    else:
        errors.append(msg)


def main():
    print("========== CHECK ALIGNED GFV OUTPUTS ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    errors = []
    passes = []

    required_files = [
        MODEL_FILE,
        SCALER_FILE,
        CONFIG_FILE,
        METRICS_FILE,
        ALL_METRICS_FILE,
        HISTORY_FILE,
        MEASURED_FILE,
        PAIRED_MRI_FILE,
        MRI_ONLY_FILE,
        COMBINED_FILE,
        TABLE_FILE,
    ]

    for f in required_files:
        check(f.exists(), f"Required file exists: {f}", errors, passes)

    if errors:
        for e in errors:
            print("ERROR:", e)
        raise SystemExit(1)

    ckpt = torch.load(MODEL_FILE, map_location="cpu")
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    with open(SCALER_FILE, "rb") as f:
        scalers = pickle.load(f)

    metrics = pd.read_csv(METRICS_FILE)
    all_metrics = pd.read_csv(ALL_METRICS_FILE)
    history = pd.read_csv(HISTORY_FILE)
    measured = pd.read_csv(MEASURED_FILE)
    paired_mri = pd.read_csv(PAIRED_MRI_FILE)
    mri_only = pd.read_csv(MRI_ONLY_FILE)
    combined = pd.read_csv(COMBINED_FILE)

    agfv_measured = count_cols(measured, "AGFV")
    agfv_paired = count_cols(paired_mri, "AGFV")
    agfv_mri_only = count_cols(mri_only, "AGFV")
    agfv_combined = count_cols(combined, "AGFV")

    print("Measured AGFV shape:", measured.shape)
    print("Paired MRI AGFV shape:", paired_mri.shape)
    print("MRI-only AGFV shape:", mri_only.shape)
    print("Combined AGFV shape:", combined.shape)
    print("AGFV columns:", len(agfv_combined))
    print()

    check("model_state_dict" in ckpt, "Checkpoint contains model_state_dict", errors, passes)
    check("config" in ckpt, "Checkpoint contains config", errors, passes)
    check("epoch" in ckpt, "Checkpoint contains best epoch", errors, passes)
    check("selection_score" in ckpt, "Checkpoint contains selection score", errors, passes)

    check(config["model"] == "MRI-aligned molecular representation model", "Config model name is correct", errors, passes)
    check(config["best_epoch"] == ckpt["epoch"], "Config best epoch matches checkpoint", errors, passes)
    check(config["combined_subjects"] == 1508, "Config combined subjects = 1508", errors, passes)

    check(len(scalers["gfv_cols"]) == 128, "Scaler file has 128 GFV columns", errors, passes)
    check(len(scalers["mfv_cols"]) == 128, "Scaler file has 128 MFV columns", errors, passes)
    check(len(scalers["agfv_cols"]) == 128, "Scaler file has 128 AGFV columns", errors, passes)

    check(measured.shape[0] == 302, "Measured GE AGFV has 302 rows", errors, passes)
    check(paired_mri.shape[0] == 289, "Paired MRI AGFV has 289 rows", errors, passes)
    check(mri_only.shape[0] == 1206, "MRI-only AGFV has 1206 rows", errors, passes)
    check(combined.shape[0] == 1508, "Combined AGFV has 1508 rows", errors, passes)

    check(len(agfv_measured) == 128, "Measured file has 128 AGFV columns", errors, passes)
    check(len(agfv_paired) == 128, "Paired MRI file has 128 AGFV columns", errors, passes)
    check(len(agfv_mri_only) == 128, "MRI-only file has 128 AGFV columns", errors, passes)
    check(len(agfv_combined) == 128, "Combined file has 128 AGFV columns", errors, passes)

    for name, df, agfv_cols in [
        ("measured", measured, agfv_measured),
        ("paired_mri", paired_mri, agfv_paired),
        ("mri_only", mri_only, agfv_mri_only),
        ("combined", combined, agfv_combined),
    ]:
        check(df["subject_id"].duplicated().sum() == 0, f"{name} has no duplicate subject_id", errors, passes)
        check(df[agfv_cols].isna().sum().sum() == 0, f"{name} has no missing AGFV values", errors, passes)
        check(df["aligned_AD_probability"].between(0, 1).all(), f"{name} probabilities are between 0 and 1", errors, passes)

    check(set(measured["agfv_source"].unique()) == {"measured_gfv_encoder"}, "Measured source label is correct", errors, passes)
    check(set(mri_only["agfv_source"].unique()) == {"mri_mfv_encoder"}, "MRI-only source label is correct", errors, passes)

    expected_splits = {"Train", "Validation", "Test"}
    check(set(metrics["Split"].unique()) == expected_splits, "Final metrics contain Train/Validation/Test", errors, passes)

    val = metrics[metrics["Split"] == "Validation"].iloc[0]
    test = metrics[metrics["Split"] == "Test"].iloc[0]

    check(float(val["MRI-derived AGFV AUC"]) >= 0.90, "Validation MRI-derived AGFV AUC >= 0.90", errors, passes)
    check(float(test["MRI-derived AGFV balanced accuracy"]) >= 0.85, "Test MRI-derived AGFV balanced accuracy >= 0.85", errors, passes)
    check(float(test["MRI-GE alignment cosine mean"]) >= 0.60, "Test MRI-GE alignment cosine mean >= 0.60", errors, passes)

    summary = []
    summary.append("========== CHECK ALIGNED GFV OUTPUTS ==========")
    summary.append(f"Measured shape: {measured.shape}")
    summary.append(f"Paired MRI shape: {paired_mri.shape}")
    summary.append(f"MRI-only shape: {mri_only.shape}")
    summary.append(f"Combined shape: {combined.shape}")
    summary.append(f"AGFV columns: {len(agfv_combined)}")
    summary.append(f"Best epoch: {ckpt['epoch']}")
    summary.append(f"Selection score: {ckpt['selection_score']}")
    summary.append(f"Passed checks: {len(passes)}")
    summary.append(f"Failed checks: {len(errors)}")
    summary.append("")

    if errors:
        summary.append("FAILED CHECKS:")
        for e in errors:
            summary.append(f"- {e}")
    else:
        summary.append("STATUS: PASS")

    SUMMARY_FILE.write_text("\n".join(summary) + "\n")

    print("Passed checks:", len(passes))
    print("Failed checks:", len(errors))

    if errors:
        print("\nFAILED CHECKS:")
        for e in errors:
            print("ERROR:", e)
        print(f"\nSummary saved to: {SUMMARY_FILE}")
        raise SystemExit(1)

    print("\nSTATUS: PASS")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print("===============================================")


if __name__ == "__main__":
    main()
