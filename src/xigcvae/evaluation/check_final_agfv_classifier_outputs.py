from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

OUT_DIR = PROJECT_ROOT / "outputs/final_agfv_classifier"
TABLE_MAIN_DIR = PROJECT_ROOT / "outputs/tables/main"
TABLE_SUPP_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

MODEL_FILE = OUT_DIR / "final_agfv_classifier_model.pkl"
SCALER_FILE = OUT_DIR / "final_agfv_classifier_scaler.pkl"
CONFIG_FILE = OUT_DIR / "final_agfv_classifier_config.json"
METRICS_FILE = OUT_DIR / "final_agfv_classifier_metrics.csv"
ALL_MODEL_METRICS_FILE = OUT_DIR / "final_agfv_classifier_all_model_metrics.csv"
PRED_FILE = OUT_DIR / "final_agfv_classifier_predictions.csv"

MAIN_TABLE = TABLE_MAIN_DIR / "Table_2_Final_AGFV_classifier_performance.xlsx"
SUPP_TABLE = TABLE_SUPP_DIR / "Table_S16_Final_AGFV_classifier_metrics.xlsx"

SUMMARY_FILE = OUT_DIR / "check_final_agfv_classifier_outputs_summary.txt"


def check(cond: bool, msg: str, errors: list[str], passes: list[str]):
    if cond:
        passes.append(msg)
    else:
        errors.append(msg)


def main():
    print("========== CHECK FINAL AGFV CLASSIFIER OUTPUTS ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    errors = []
    passes = []

    required_files = [
        MODEL_FILE,
        SCALER_FILE,
        CONFIG_FILE,
        METRICS_FILE,
        ALL_MODEL_METRICS_FILE,
        PRED_FILE,
        MAIN_TABLE,
        SUPP_TABLE,
    ]

    for f in required_files:
        check(f.exists(), f"Required file exists: {f}", errors, passes)

    if errors:
        for e in errors:
            print("ERROR:", e)
        raise SystemExit(1)

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_FILE, "rb") as f:
        scaler_obj = pickle.load(f)

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    metrics = pd.read_csv(METRICS_FILE)
    all_models = pd.read_csv(ALL_MODEL_METRICS_FILE)
    preds = pd.read_csv(PRED_FILE)

    check(hasattr(model, "predict_proba"), "Model has predict_proba", errors, passes)
    check("scaler" in scaler_obj, "Scaler object contains scaler", errors, passes)
    check("agfv_cols" in scaler_obj, "Scaler object contains AGFV columns", errors, passes)
    check(len(scaler_obj["agfv_cols"]) == 128, "Scaler object has 128 AGFV columns", errors, passes)

    check(config["model"] == "Final AGFV logistic classifier", "Config model name is correct", errors, passes)
    check(config["n_subjects"] == 1508, "Config subject count = 1508", errors, passes)
    check(config["n_agfv_features"] == 128, "Config AGFV feature count = 128", errors, passes)

    check(preds.shape[0] == 1508, "Prediction file has 1508 rows", errors, passes)
    check(preds["subject_id"].duplicated().sum() == 0, "Prediction file has no duplicate subject_id", errors, passes)

    prob_col = "final_agfv_classifier_probability_AD"
    pred_col = "final_agfv_classifier_prediction"

    check(prob_col in preds.columns, "Prediction probability column exists", errors, passes)
    check(pred_col in preds.columns, "Prediction label column exists", errors, passes)

    if prob_col in preds.columns:
        check(preds[prob_col].between(0, 1).all(), "Predicted probabilities are between 0 and 1", errors, passes)

    if pred_col in preds.columns:
        check(set(preds[pred_col].unique()).issubset({0, 1}), "Predicted labels are 0/1", errors, passes)

    expected_splits = {"Train", "Validation", "Test"}
    check(expected_splits.issubset(set(metrics["Split"].unique())), "Metrics contain Train/Validation/Test", errors, passes)

    overall = metrics[metrics["Subgroup"] == "Overall"]
    check(overall.shape[0] == 3, "Overall metrics contain three split rows", errors, passes)

    check(all_models.shape[0] >= 5, "C-grid model metrics are present", errors, passes)

    test_overall = overall[overall["Split"] == "Test"]
    check(test_overall.shape[0] == 1, "One overall test row exists", errors, passes)

    summary = []
    summary.append("========== CHECK FINAL AGFV CLASSIFIER OUTPUTS ==========")
    summary.append(f"Predictions shape: {preds.shape}")
    summary.append(f"Metrics shape: {metrics.shape}")
    summary.append(f"All model metrics shape: {all_models.shape}")
    summary.append(f"Best C: {config['best_C']}")
    summary.append("")
    summary.append("Overall metrics:")
    summary.append(overall.to_string(index=False))
    summary.append("")
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
    print("==========================================================")


if __name__ == "__main__":
    main()
